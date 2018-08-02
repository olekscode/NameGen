# Based on: https://gist.github.com/macieksk/038b201a54d9e804d1b5

import os
import datetime
from io import StringIO

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class Drive:
    def __init__(self, folder_name):
        self.__login()
        self.__folder = self.__find_folders(folder_name)[0]

        self.__files = {}
        self.__logs = {}


    def log(self, string, fname, append=True):
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
            self.__files[fname] = self.__create_text_file(fname)
            self.__logs[fname] = []

        if not append:
            # Clear the log
            self.__logs[fname] = []

        string = '[{}]\n{}\n'.format(datetime.datetime.now(), string)

        self.__logs[fname].append(string)
        self.__files[fname].SetContentString('\n'.join(self.__logs[fname]))
        self.__files[fname].Upload()


    def log_dataframe(self, df, fname, append=False):
        s = StringIO()
        df.to_csv(s)
        string = s.getvalue()

        self.log(string, fname, append=append)


    def upload_image(self, fname, update=True):
        if fname not in self.__files.keys() or not update:
            self.__files[fname] = self.__create_image_file(
                os.path.basename(fname))

        self.__files[fname].SetContentFile(fname)
        self.__files[fname].Upload()


    def __login(self):
        self.__gauth = GoogleAuth()
        self.__gauth.LocalWebserverAuth()        # Creates local webserver and auto handles authentication
        self.__drive = GoogleDrive(self.__gauth) # Create GoogleDrive instance with authenticated GoogleAuth instance


    def __find_folders(self, fldname):
        file_list = self.__drive.ListFile({
            'q': "title='{}' and mimeType contains 'application/vnd.google-apps.folder' and trashed=false".format(fldname)
            }).GetList()
        return file_list


    def __create_file(self, name, mime_type=None):
        param = {
            'title': name,
            'parents': [{ u'id': self.__folder['id'] }]
        }

        if mime_type != None:
            param['mimeType'] = mime_type

        return self.__drive.CreateFile(param)


    def __create_text_file(self, name):
        return self.__create_file(name)


    def __create_csv_file(self, name):
        return self.__create_file(name, 'text/csv')


    def __create_image_file(self, name):
        return self.__create_file(name)