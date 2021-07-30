import matplotlib.pyplot as plt
from numerapi.numerapi import NumerAPI
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from numebot.data.data_constants import NC
from numebot.data.data_manager import DataManager
from numebot.file_names_getter import FileNamesGetter
from numebot.models.numerai_model import NumeraiModel

from numebot_private.data.data_manager_extended import DataManagerExtended

data_type_colormap = {'train': "blue", 
                      'validation': "orange",
                      'validation_1': "orange",
                      'validation_2': "lightgreen",
                      'test': "red"}

class TrainByEraSimilarity(NumeraiModel):

    def __init__(
        self, config_row: pd.Series, file_names: FileNamesGetter, napi: NumerAPI,
        data: DataManager, k_nearest=20,
        testing: bool=False
    ):

        class_name = str(self.__class__).rstrip('>\'').split('.')[-1]
        print(f'Creating {class_name}')

        super().__init__(config_row=config_row, file_names=file_names, napi=napi, testing=testing)

        self.folder = self.names.model_folder(self.name)/'other_files'
        name = 'pca_train_val'
        self.projector_path = self.folder/f'{name}_pca_fitted.joblib'
        self.scaler_path = self.folder/f'{name}_scaler.joblib' 

        self.k = k_nearest

        self.model_ready = True

    def load_model(self):
        return None

    def predict(self, data: DataManager, to_be_saved_for_submission=False):
        """
        Overwrite the general version of predict.
        """
        numerai_data_set = data.tournament
        
        self.data = DataManagerExtended(
            file_names=data.names, nrows=data.nrows, save_memory=data.save_memory
        )

        print('Training era similarity')
        self.projector = self.get_data_projector(self.data)
        self.projected_rows = self.project_data(self.data.train_val, self.data.features)
        self.eras, self.scaler = self.get_eras_representations(self.projected_rows)

        self.live_projected = self.project_data(self.data.live, self.data.features)
        self.live_era, _ = self.get_eras_representations(self.live_projected, scaler=self.scaler)

        self.nearest_eras = self.find_nearest_eras(self.eras, target=self.live_era, k=self.k)

        train_eras = list(self.nearest_eras.index.get_level_values(0))
        train_data = self.data.train_val.loc[self.data.train_val[NC.era].isin(train_eras)]

        self.train_model(dataset=train_data)
    
        if not self.model_ready:
            print(f'Model {self.name} is not ready, it needs to be trained or loaded.')
            return None

        print(f'\nRunning prediction for model {self.name} ...')
        print(f' - Rows: {len(numerai_data_set)}, columns: {len(numerai_data_set.columns)}')
        
        feature_cols = [f for f in numerai_data_set.columns if f.startswith("feature")]
        
        output_values = self.model.predict(numerai_data_set[feature_cols])
        output = pd.DataFrame({'prediction': output_values}, index=numerai_data_set.index)

        if to_be_saved_for_submission and not self.testing:
            self.save_for_submission(output)
            
        return output

    def get_data_projector(self, data):
        print('Getting data projector')
        pca = PCA(n_components=2)
        pca.fit(data.train_val[data.features])

        self.projector_path.parent.mkdir(exist_ok=True, parents=True)
        dump(pca, self.projector_path)

        return pca

    def project_data(self, dataset, features):
        print('Projecting data')
        projected = self.projector.transform(dataset[features])
        projected = pd.DataFrame(projected, index=dataset.index)

        projected = pd.DataFrame({'era': dataset[NC.era],
                                  'data_type': dataset[NC.data_type],
                                  'trans_0': projected[0],
                                  'trans_1': projected[1]})
    
        return projected

    def get_eras_representations(self, projected, scaler=None):
        print('Getting eras representation, scaled.')
        eras = projected.groupby([NC.era,NC.data_type]).agg(['mean', 'std'])
        
        if scaler is None:
            scaler = StandardScaler()   
            scaler.fit(eras)

        eras_scaled = pd.DataFrame(scaler.transform(eras), 
                                   index=eras.index,
                                   columns=eras.columns)

        print(f'Full training set: {len(eras)} eras.')

        return eras_scaled, scaler

    def plot_projected_rows(self, fraction=0.01):

        indexes = self.data.train_val.sample(frac=fraction).index

        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        ax.scatter(
            self.projected_rows.loc[indexes]['trans_0'],
            self.projected_rows.loc[indexes]['trans_1'],
            c=self.data.train_val.loc[indexes][NC.data_type].map(data_type_colormap),
            alpha=0.1,
        )

        return ax

    def plot_eras_representation(self, other_points=None):

        c = self.eras.index.get_level_values(0)
        c = np.select(
            [c <= 120, (c > 120) & (c <= 132), (c > 132)],
            ['train', 'validation_1', 'validation_2'],
            c
        )
        c = pd.Series(c)

        plt.figure(figsize=(15,8))
        axes = {}
        axes['mean'] = plt.subplot(1,2,1)
        axes['mean'].scatter(
            self.eras[('trans_0', 'mean')],
            self.eras[('trans_1', 'mean')],
            c=c.map(data_type_colormap),
        );
        axes['mean'].scatter(
            self.live_era[('trans_0', 'mean')],
            self.live_era[('trans_1', 'mean')],
            c='red',
        );

        axes['std'] = plt.subplot(1,2,2)
        axes['std'].scatter(
            self.eras[('trans_0', 'std')],
            self.eras[('trans_1', 'std')],
            c=c.map(data_type_colormap),
        );
        axes['std'].scatter(
            self.live_era[('trans_0', 'std')],
            self.live_era[('trans_1', 'std')],
            c='red',
        );

        if other_points is not None:
            axes['mean'].plot(
                other_points[('trans_0', 'mean')],
                other_points[('trans_1', 'mean')],
                'o',
                markerfacecolor='none',
                markeredgecolor='red',
                markersize=15,
            );

            axes['std'].plot(
                other_points[('trans_0', 'std')],
                other_points[('trans_1', 'std')],
                'o',
                markerfacecolor='none',
                markeredgecolor='red',
                markersize=15,
            );

        return axes

    def find_nearest_eras(self, eras, target, k):
        cols = [
            ('trans_0', 'mean'),
            ('trans_1', 'mean'),
            ('trans_0', 'std'),
            ('trans_1', 'std'),
        ]
        
        differences_squared = [(eras[col].values - target[col].values)**2 for col in cols]
        distances = np.sqrt(np.sum(differences_squared, axis=0))

        eras['distance'] = distances

        k_nearest_eras = eras.sort_values('distance').iloc[:k]

        return k_nearest_eras

    def train_model(self, dataset):
        print('Training set:  ', dataset.shape)
        print(f"Training model on {self.k} nearest eras...")
        
        self.model = XGBRegressor(max_depth=5, 
                                  learning_rate=0.01,
                                  n_estimators=2000,
                                  n_jobs=-1,
                                  colsample_bytree=0.1)

        # Load the example model trained on train + val
        model_file = self.names.model_path('rpica_test_2', suffix='xgb')
        self.model.load_model(model_file)
        self.model.n_estimators +=1000

        self.model.fit(dataset[self.data.features], dataset[NC.target])

        # if not self.testing:
        #     self.save_model()
        # else:
        #     print('Testing mode: trained model not saved.')
        print('Not saving model: changes per era')

        self.model_ready = True
        print('Training finished!')
