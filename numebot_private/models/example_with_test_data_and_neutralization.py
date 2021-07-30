from numebot_private.data.data_manager_extended import DataManagerExtended
from numebot_private.models.example_with_test_data import ExampleWithTestData


# Inherits from ExampleModel!
class ExampleWithTestDataAndNeutralization(ExampleWithTestData):

    def train_model(self, data: DataManagerExtended):
        print('Original test set:  ', data.test.shape)
        print("Training model...")
        
        feature_names = [f for f in data.test.columns if f.startswith("feature")]
        self.model.fit(data.test[feature_names], data.test['target'])

        self.save_model()
        self.model_ready = True
        
        print('Training finished!')
