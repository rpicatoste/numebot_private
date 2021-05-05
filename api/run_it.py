from numebot_private.round_manager_extended import RoundManagerExtended

from numebot.secret import PUBLIC_ID, SECRET_KEY


data_folder = '/home/pica/nas_pica/Data/numerai/'

print('\nRunning numebot private\n')

rm = RoundManagerExtended(
    data_folder, 
    public_id=PUBLIC_ID, 
    secret_key=SECRET_KEY,
#    nrows=10000, testing=True,
)

rm.models_info()
rm.generate_predictions_for_all_models()
rm.submit_predictions()
