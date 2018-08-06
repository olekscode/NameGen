import constants

# TODO: Move elsewhere
from metrics import confusion_dataframe
from visualizations import plot_confusion_dataframe, plot_history, COLORS

import os
import datetime
import pickle
from io import StringIO

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import numpy as np
import matplotlib.pyplot as plt


class DefaultLogger:
    def __init__(self):
        self.log_files = []


    def write_log(self, string, fname, append):
        self.remove_old_file_on_first_log(fname)

        with open(fname, 'a') as f:
            f.write(string)

    
    def log(self, string, fname=os.path.join(constants.LOGS_DIR, 'log.txt'), append=True):
        string = '[{}]\n{}\n\n'.format(datetime.datetime.now(), string)
        self.write_log(string, fname, append)

    
    def save_dataframe(self, df, fname):
        df.to_csv(fname, sep='\t')
        self.on_file_saved(fname)


    def save_image(self, fig, fname, update=True):
        fig.savefig(fname)
        plt.close(fig)
        self.on_file_saved(fname)


    def save_pickle(self, obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

        self.on_file_saved(fname)


    def on_file_saved(self, fname):
        """DefaultLogger foes nothing, DriveLogger uploads file to Drive"""
        pass

    
    def write_training_log(self, log_dict, fname):
        log_template = ": {}\n".join(log_dict.keys()) + ": {}"
        log_string = log_template.format(*log_dict.values())

        self.log(log_string, fname=fname, append=True)

    # TODO: Refactor
    def plot_and_save_histories(self, loss_history, bleu_history, rouge_history, f1_history, num_unique_names_history):
        fig = plot_history(
            history = loss_history,
            color = COLORS['red'],
            title = 'Average Loss',
            ylabel = 'Loss')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'loss.png'))

        fig = plot_history(
            history = bleu_history,
            color = COLORS['blue'],
            title = 'Average BLEU',
            ylabel = 'BLEU')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'bleu.png'))

        fig = plot_history(
            history = rouge_history,
            color = COLORS['green'],
            title = 'Average ROUGE',
            ylabel = 'ROUGE')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'rouge.png'))

        fig = plot_history(
            history = f1_history,
            color = COLORS['red'],
            title = 'Average F1 score',
            ylabel = 'F1 score')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'f1.png'))

        # TODO: Move to visualizations.py
        fig, ax = plt.subplots()
        x = constants.LOG_EVERY * np.arange(len(bleu_history))
        ax.plot(x, bleu_history, COLORS['blue'])
        ax.plot(x, rouge_history, COLORS['green'])
        ax.plot(x, f1_history, COLORS['red'])
        ax.set_title('Average BLEU, ROUGE, and F1 hiestories')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'bleu_rouge_f1.png'))

        fig = plot_history(
            history = num_unique_names_history,
            color = COLORS['yellow'],
            title = 'Number of unique names',
            ylabel = '# names')

        self.save_image(fig, os.path.join(constants.IMG_DIR, 'num_names.png'))


    def remove_old_file_on_first_log(self, fname):
        # Is this a first log?
        if fname not in self.log_files:
            self.log_files.append(fname)

            try:
                # Remove if exists
                os.remove(fname)
            except OSError:
                # Otherwise do nothing
                pass


class DriveLogger(DefaultLogger):
    """Inspired by: https://gist.github.com/macieksk/038b201a54d9e804d1b5"""

    def __init__(self, folder_name):
        super(DriveLogger, self).__init__()
        
        self.__login()
        self.__folder = self.__find_folders(folder_name)[0]

        self.__files = {}
        self.__logs = {}


    def write_log(self, string, fname, append):
        """Writes a log string to the file on Google Drive.

        If file with this name doesn't exist it gets created.

        Parameters
        ----------

        string : str
            A string to be written

        fname : str
            File name on Google Drive

        append : bool
            Append to the end of existing file or rewrite it

        Examples
        --------

        >>> from drive import Drive

        Login and authenticate.

        >>> drive = Drive('NameGen')

        Create a new file and write several logs to it.

        >>> drive.log('Lorem ipsum', 'lipsum.txt')
        >>> drive.log('dolor sit', 'lipsum.txt')
        >>> drive.log('amet', 'lipsum.txt')

        Create a new file and override it with each log

        >>> drive.log('Hello', 'hello.txt')
        >>> drive.log('world', 'hello.txt', append=False)
        """

        super(DriveLogger, self).write_log(string, fname, append)
        fname = os.path.basename(fname)

        if fname not in self.__files.keys():
            self.__files[fname] = self.__create_file(fname)
            self.__logs[fname] = []

        if not append:
            # Clear the log
            self.__logs[fname] = []

        self.__logs[fname].append(string)
        self.__files[fname].SetContentString(''.join(self.__logs[fname]))
        self.__files[fname].Upload()


    def on_file_saved(self, fname):
        self.__upload_file(fname, update=True)


    def __login(self):
        self.__gauth = GoogleAuth()
        self.__gauth.LocalWebserverAuth()        # Creates local webserver and auto handles authentication
        self.__drive = GoogleDrive(self.__gauth) # Create GoogleDrive instance with authenticated GoogleAuth instance


    def __find_folders(self, fldname):
        file_list = self.__drive.ListFile({
            'q': "title='{}' and mimeType contains 'application/vnd.google-apps.folder' and trashed=false".format(fldname)
            }).GetList()
        return file_list


    def __create_file(self, name):
        param = {
            'title': name,
            'parents': [{ u'id': self.__folder['id'] }]
        }

        return self.__drive.CreateFile(param)

    
    def __upload_file(self, fname, update=True):
        if fname not in self.__files.keys() or not update:
            self.__files[fname] = self.__create_file(
                os.path.basename(fname))

        self.__files[fname].SetContentFile(fname)
        self.__files[fname].Upload()