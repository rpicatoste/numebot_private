from numebot_private.round_manager_extended import RoundManagerExtended

from numebot.env import NUMERAI_DATA_FOLDER
from numebot.secret import PUBLIC_ID, SECRET_KEY


print('\nRunning numebot private\n')

rm = RoundManagerExtended(
    NUMERAI_DATA_FOLDER, 
    public_id=PUBLIC_ID, 
    secret_key=SECRET_KEY,
#    nrows=10000, testing=True,
)

rm.models_info()
rm.generate_predictions_for_all_models()
rm.submit_predictions()
_ = rm.mm.download_round_details()
