import os
from wsgiref.simple_server import WSGIServer

import numpy as np
import pandas as pd
import json
from flask import Flask, request, send_from_directory, make_response
from flask_restful import Api, reqparse, abort
from werkzeug.utils import secure_filename

import FlaskRegNacos
import sys,signal,time
import AppConfig
import views
from inference_service import InferenceService
from test import Test
from parameter import Parameter
from option import args
from views import *

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['csv'])

p = Parameter()

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True
api = Api(app)


api.add_resource(DataSet, '/dataset')

@app.route('/sharkms/<string:function>/<string:model_name>/<int:userId>/<int:logonSequence>/<string:vcode>', methods=['POST'])
def model(function,model_name,userId, logonSequence, vcode):
    resData = {
        "servicer": AppConfig.servicer,
        "startTime": int(time.time() * 1000)
    }
    if(function == 'model' and model_name == 'predict'):
        inference = InferenceService.INSTANCE
        if inference is None:
            resData.setdefault('state', 505)
            resData.setdefault('message', "尚未训练模型")
            response = make_response(json.dumps(resData, separators=(',', ':'), ensure_ascii=False))
            response.mimetype = 'application/json;charset=utf-8'
            return response, 200
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('data', type=dict, required=True)
        data = parser.parse_args()
        ret = dict()
        ret['model'] = inference.get_model_name()
        try:
            ret['prediction'] = inference.get_prediction(data)
            resData.setdefault('data', ret)
            resData.setdefault('state', 1)
        except Exception:
            resData.setdefault('state', 504)
            resData.setdefault('message', "参数错误")
    elif(function == 'model' and model_name == 'xgboost'):
        inference = InferenceService().INSTANCE
        parser = reqparse.RequestParser(bundle_errors=True)
        # 主要参数
        parser.add_argument('dataset', required=True)
        parser.add_argument('parameter', required=True)
        data = parser.parse_args()
        # 向接口传训练参数
        detail = dict()
        try:
            inference.solve_dataset(data['dataset'])
            detail = inference.train_model(model_name, data['parameter'])
            resData.setdefault('data', detail)
            resData.setdefault('state', 1)
        except FileNotFoundError:
            resData.setdefault('state', 501)
            resData.setdefault('message', "数据集不存在")
        except ModuleNotFoundError:
            resData.setdefault('state', 502)
            resData.setdefault('message', "模型名错误")
    elif(function == 'model' and model_name == 'upload'):
        path = os.path.join(AppConfig.UPLOAD_FOLDER, 'test.csv')
        if os.path.exists(path):
            os.remove(path)
        f = request.files.get('file')
        temp_path = os.path.join(AppConfig.UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(temp_path)
        os.rename(temp_path, path)
        resData.setdefault('state', 1)
        resData.setdefault('message', "成功上传")
        if f is None:
            resData.setdefault('state', 507)
            resData.setdefault('message', "没有上传图片，请重新上传！")
        if not allow_file(f.filename):
            resData.setdefault('state', 507)
            resData.setdefault('message', "文件格式不支持，请上传CSV文件！")
    elif(function == 'model' and model_name == 'set'):
        target_len = request.json.get('target_len')
        predict_column = request.json.get('predict_column')
        path = os.path.join(AppConfig.UPLOAD_FOLDER, 'test.csv')
        data = pd.read_csv(path)
        exist = False
        for col in data.columns:
            if (col == predict_column):
                exist = True
                break
        if not exist:
            resData.setdefault('state', 508)
            resData.setdefault('message', "不存在当前列")
        else:
            p.set_predict_column(predict_column)
            p.set_test_target_len(target_len)
            if target_len and predict_column:
                resData.setdefault('state', 1)
                resData.setdefault('message', "设置成功")
            else:
                resData.setdefault('state', 509)
                resData.setdefault('message', "设置错误")
    elif(function == 'model' and model_name == 'download'):
        test = Test()
        test.set(test_target_len=p.get_test_target_len(), predict_column=p.get_predict_column())
        test.run()
        filename = 'result.csv'
        path = os.path.isfile(os.path.join(AppConfig.DOWNLOAD_FOLDER, filename));
        if path:
            return send_from_directory(directory=AppConfig.DOWNLOAD_FOLDER, path=filename, as_attachment=True)
            resData.setdefault('data', send_from_directory(directory=AppConfig.DOWNLOAD_FOLDER, path=filename, as_attachment=True))
            resData.setdefault('state', 1)
            # response = make_response(json.dumps(resData, separators=(',', ':'), ensure_ascii=False))
            # response.mimetype = 'application/json;charset=utf-8'
            return resData
        else:
            resData.setdefault('state', 510)
            resData.setdefault('message', "发生错误")
    else:
        resData.setdefault('state', 503)
        resData.setdefault('message', "方法错误")
    response = make_response(json.dumps(resData,  separators=(',', ':'), ensure_ascii=False))
    response.mimetype = 'application/json;charset=utf-8'
    return response, 200




@app.route("/download")
def download():
    test = Test()
    test.set(test_target_len=p.get_test_target_len(),predict_column=p.get_predict_column())
    test.run()
    filename = 'result.csv'
    path = os.path.isfile(os.path.join(app.config['DOWNLOAD_FOLDER'], filename));
    if path:
        return send_from_directory(directory=app.config['DOWNLOAD_FOLDER'], path=filename, as_attachment=True)
    ren = {'msg': '发生错误', 'msg_code': -1}
    return json.dumps(ren, ensure_ascii=False)

def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#
# def beforeExit(signum, frame):
# 	FlaskRegNacos.down()
# 	print( "等待instanceDown执行结束...")
# 	sys.exit()
#
# #注册信号处理
# signal.signal(signal.SIGINT, beforeExit)
# signal.signal(signal.SIGHUP, beforeExit)
# signal.signal(signal.SIGTERM, beforeExit)
#
# upState = FlaskRegNacos.up({
# 		'serviceName': AppConfig.servicer,
# 		'ip': AppConfig.appAddr,
# 		'port': AppConfig.appPort,
# 		'weight': 1.0,
# 		'ephemeral': True
# 	})
# if( upState is False):
# 	print( "注册nacos2失败，退出服务！")
# 	sys.exit()

if __name__ == "__main__":
    app.run(debug = True)
    http_server = WSGIServer(('', AppConfig.appPort), app)
    http_server.serve_forever()
