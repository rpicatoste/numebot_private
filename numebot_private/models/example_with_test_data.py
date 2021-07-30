from numebot.models.example_model import ExampleModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class ExampleWithTestData(ExampleModel):

    def train_model(self, data: DataManagerExtended):
        print('Original test set:  ', data.test.shape)
        print("Training model...")
        
        feature_names = [f for f in data.test.columns if f.startswith("feature")]
        self.model.fit(data.test[feature_names], data.test['target'])

        self.save_model()
        print('Training finished!')
        
        self.model_ready = True
