from numebot.models.example_model import ExampleModel

from numebot_private.data.data_manager_extended import DataManagerExtended


# Inherits from ExampleModel!
class ExampleModelNoisyData(ExampleModel):

    def train_model(self, data: DataManagerExtended, N=10):
        print(f'Generating dataset {N} times bigger with noise ...')
        with_noise = data.noisy_training(N)
        print('Original train set:  ', data.train.shape)
        print('With noise train set:', with_noise.shape)

        feature_names = [f for f in with_noise.columns if f.startswith("feature")]
        print("Training model...")
        self.model.fit(with_noise[feature_names], with_noise['target'])

        self.save_model()
        self.model_ready = True
        
        print('Training finished!')
