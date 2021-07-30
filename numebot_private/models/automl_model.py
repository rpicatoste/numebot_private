from numebot.models.numerai_model import NumeraiModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class AutomlModel(NumeraiModel):

    def load_model(self):
        class_name = str(self.__class__).rstrip('>\'').split('.')[-1]
        print(f'Creating {class_name}')
        model_folder = self.names.model_path(self.name, suffix='')

        # This is the model that generates the included example predictions file.
        # Taking too long? Set learning_rate=0.1 and n_estimators=200 to make this run faster.
        # Remember to delete example_model.xgb if you change any of the parameters below.
        model = XGBRegressor(max_depth=5, 
                             learning_rate=0.01, 
                             n_estimators=2000, 
                             n_jobs=-1, 
                             colsample_bytree=0.1)

        if model_file.exists():
            print("Loading pre-trained model...")
            model.load_model(model_file)
            self.model_ready = True
        else:
            print(f'WARNING: Model for {self.name} is not trained: Run train!')
    
        return model

    def train_model(self, data: DataManagerExtended, N=10):
        print(f'Generating dataset {N} times bigger with noise ...')
        with_noise = data.noisy_training(N)
        print('Original train set:  ', data.train.shape)
        print('With noise train set:', with_noise.shape)

        feature_names = [f for f in with_noise.columns if f.startswith("feature")]
        print("Training model...")
        self.model.fit(with_noise[feature_names], with_noise['target'])

        self.save_model()
        print('Training finished!')
