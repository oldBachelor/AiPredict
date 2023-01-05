import requests
import json,time
from threading import Timer
import AppConfig

nacosURL = AppConfig.nacosUrl
globalInstanceParams={}
globalTimerBeat = True
requestTimeout = (5, 60)

def instanceUp( params):
	dataParams = params.copy()
	dataParams.update({
		'enabled': True,
		'healthy': True
	})
	try:
		r = requests.post(nacosURL+'/v1/ns/instance', data=dataParams, timeout=requestTimeout)
		if( r.status_code==200): return True;
		print( 'instanceUp code={}, text={}'.format( r.status_code, r.text))
	except Exception as err:
		print( 'instanceUp err={}'.format( err))
	return False;

def instanceDown( params):
	try:
		r = requests.delete(nacosURL+'/v1/ns/instance', data=params, timeout=requestTimeout)
		if( r.status_code==200): return True;
		print( 'instanceDown code={}, text={}'.format( r.status_code, r.text))
	except Exception as err:
		print( 'instanceDown err={}'.format( err))
	return False;

def instanceInfo( params):
	try:
		r = requests.get(nacosURL+'/v1/ns/instance', params=params, timeout=requestTimeout)
		if( r.status_code==200):
			rjson = r.json()
			return {
				'serviceName': params.get( 'serviceName'), 
				'ephemeral': params.get( 'ephemeral'), 
				'beat': json.dumps({
					'serviceName': params.get( 'serviceName'),
					'ip': params.get( 'ip'), 
					'port': params.get( 'port'),
					"cluster": rjson.get("clusterName"),
					"weight": params.get( 'weight'),
					"metadata":{},
					"scheduled": False,
					"period": 20,
					"stopped": False
				})
			}
		print( 'instanceInfo code={}, text={}'.format( r.status_code, r.text))
	except Exception as err:
		print( 'instanceInfo err={}'.format( err))
	return None;

def instanceBeat( params):
	if( globalTimerBeat is False): return
	beatInterval = 2
	try:
		#参数应该不全，不能使用lightBeatEnabled模式；查源代码至少差ip,port等等
		r = requests.put(nacosURL+'/v1/ns/instance/beat', data=params, timeout=requestTimeout)
		if( r.status_code==200):
			beatInterval = int( r.json().get('clientBeatInterval',3000) / 1000)
			#print( 'instanceBeat ok, next {}s, data={}'.format( beatInterval, r.text))
		else:
			print( 'instanceBeat code={}, text={}'.format( r.status_code, r.text))
		if( r.status_code==404):
			#重新注册
			print( 'instanceBeat try instanceUp res={}'.format( instanceUp( globalInstanceParams)))
	except Exception as err:
		print( 'instanceBeat err={}'.format( err))
	Timer( beatInterval, instanceBeat, args=(params,)).start() #必须在params后加,
		
def up( params):
	#上线实例
	instanceUp( params)
	#获取实例
	time.sleep(1)
	beatParams = instanceInfo( params)
	if( beatParams is None): return False
	#定时心跳
	print( 'instanceBeat begin {}'.format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
	instanceBeat( beatParams)
	#设置变量，用于下线不再传递参数
	global globalInstanceParams
	globalInstanceParams = params
	return True

def down():
	global globalTimerBeat
	globalTimerBeat = False
	instanceDown( globalInstanceParams)

