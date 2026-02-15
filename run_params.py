
import yaml
import os
class RunParameters():

    def __init__(self, folder):

        self.run_params={}
        self.model_params={}
        self.data_params={}

        self.run_params['Data']=self.data_params
        self.run_params['Model']=self.model_params
        self.yaml_folder=folder

    def save_to_yaml(self):
        if not os.path.exists(self.yaml_folder):
            os.makedirs(self.yaml_folder)
        
        with open(self.yaml_folder+'run_parameters.yaml', 'w+') as file:
            yaml.dump(self.run_params, file)
    

    def update_model_params(self,key,value,parent_key=None,):
        if parent_key==None:

            self.model_params[key]=value
        else:
            if parent_key not in self.model_params:
                self.model_params[parent_key]={}
             
            self.model_params[parent_key][key]=value

    def update_data_params(self, key,value,parent_key=None):
        if parent_key==None:
            self.data_params[key]=value
        else:
            if parent_key not in self.data_params:
                self.data_params[parent_key]={}

            self.data_params[parent_key][key]=value