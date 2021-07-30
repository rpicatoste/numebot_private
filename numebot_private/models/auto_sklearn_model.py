import pickle

from numebot.models.numerai_model import NumeraiModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class AutoSklearnModel(NumeraiModel):

    def load_model(self):
        class_name = str(self.__class__).rstrip('>\'').split('.')[-1]
        print(f'Creating {class_name}')
        model_path = self.names.model_path(self.name, 
                                           suffix='pkl')

        if model_path.exists():
            print("Loading pre-trained model...")
            model = pickle.load(open(model_path,'rb'))
            self.model_ready = True
        else:
            print(f'WARNING: Model for {self.name} is not trained: Run train!')
    
        return model

    def train_model(self, data: DataManagerExtended):
        pass