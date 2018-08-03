import os
import datetime
from io import StringIO
from abc import ABC, abstractmethod

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# TODO: Move elsewhere
from metrics import confusion_dataframe
from visualizations import plot_confusion_dataframe, plot_history, COLORS


class AbstractLogger(ABC):
    @abstractmethod
    def log_write(self, string, fname, append):
        pass

    
    def log(self, string, fname="log.txt", append=True):
        string = '[{}]\n{}\n'.format(datetime.datetime.now(), string)
        self.log_write(string, fname, append)

    
    def log_dataframe(self, df, fname="log.txt", append=True):
        s = StringIO()
        df.to_csv(s)
        string = s.getvalue()

        self.log(string, fname, append)


    def log_image(self, fig, fname, update=True):
        fig.savefig(fname)

    
    def write_training_log(self, log_dict, bleu_history, rouge_history, loss_history, names, confusion_df):
        log_template = ": {}\n".join(log_dict.keys()) + ": {}"
        log_string = log_template.format(*log_dict.values())

        self.log(log_string, fname="train-log.txt")

        fig = plot_history(
            history = loss_history,
            color = COLORS['red'],
            title = 'Average Loss',
            ylabel = 'Loss')

        self.log_image(fig, '../img/loss.png')

        fig = plot_history(
            history = bleu_history,
            color = COLORS['blue'],
            title = 'Average BLEU',
            ylabel = 'BLEU')

        self.log_image(fig, '../img/bleu.png')

        fig = plot_history(
            history = rouge_history,
            color = COLORS['green'],
            title = 'Average ROUGE',
            ylabel = 'ROUGE')

        self.log_image(fig, '../img/rouge.png')

        fig = plot_confusion_dataframe(confusion_df)
        self.log_image(fig, '../img/confusion.png')

        self.log_dataframe(names.sort_values('BLEU', ascending=False).head(20), 'names.txt', append=False)
        self.log_dataframe(confusion_df[confusion_df['PP'] > 0]['PP'], 'unique_names.txt')


class ConsoleLogger(AbstractLogger):
    def log_write(self, string, fname, append):
        print(string)


class DriveLogger(AbstractLogger):
    """Inspired by: https://gist.github.com/macieksk/038b201a54d9e804d1b5"""

    def __init__(self, folder_name):
        self.__login()
        self.__folder = self.__find_folders(folder_name)[0]

        self.__files = {}
        self.__logs = {}


    def log_write(self, string, fname, append):
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

        if fname not in self.__files.keys():
            self.__files[fname] = self.__create_file(fname)
            self.__logs[fname] = []

        if not append:
            # Clear the log
            self.__logs[fname] = []

        self.__logs[fname].append(string)
        self.__files[fname].SetContentString('\n'.join(self.__logs[fname]))
        self.__files[fname].Upload()


    def log_dataframe(self, df, fname="log.txt", append=True):
        s = StringIO()
        df.to_csv(s)
        string = s.getvalue()

        self.log(string, fname, append=append)

    
    def log_image(self, fig, fname, update=True):
        super(DriveLogger, self).log_image(fig, fname, update)
        self.__upload_file(fname, update)
        os.remove(fname)


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