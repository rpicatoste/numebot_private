from numebot.round_manager import RoundManager

from numebot_private.data.data_manager_extended import DataManagerExtended

class RoundManagerExtended(RoundManager):

    def load_data_manager(self, nrows, save_memory):
        """
        A function is used to load the data manager so it can be overriden by a parent class of 
        RoundManager.
        """
        return DataManagerExtended(file_names=self.names, nrows=nrows, save_memory=save_memory)
