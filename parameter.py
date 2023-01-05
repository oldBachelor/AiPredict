class Parameter():
    def __init__(self):
        self.test_target_len=120
        self.predict_column= 'Sunspots'
    def set_test_target_len(self,test_target_len):
        self.test_target_len = test_target_len
    def set_predict_column(self,  predict_column):
        self.predict_column = predict_column
    def get_test_target_len(self):
        return self.test_target_len
    def get_predict_column(self):
        return self.predict_column