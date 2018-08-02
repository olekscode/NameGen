# Based on: https://gist.github.com/macieksk/038b201a54d9e804d1b5

import os
from io import StringIO
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class DriveUploader:
    def __init__(self, folder_name):
        self.__login()
        self.__folder = self.__find_folders(folder_name)[0]

        self.__log = []

        self.__log_file = self.__drive.CreateFile({
            'title':'log.txt',
            'parents':[{ u'id': self.__folder['id'] }]
        })

        self.__translations_file = self.__drive.CreateFile({
            'title':'translations.csv',
            'mimeType':'text/csv',
            'parents':[{ u'id': self.__folder['id'] }]
        })

    def upload_to_drive(self, files):
        self.__upload_files_to_folder(files, self.__folder)

    
    def log(self, string):
        self.__log.append(string)
        self.__log_file.SetContentString('\n'.join(self.__log))
        self.__log_file.Upload()

    
    def update_translations(self, df):
        s = StringIO()
        df.to_csv(s)
        self.__translations_file.SetContentString(s.getvalue())
        self.__translations_file.Upload()


    def __login(self):
        self.__gauth = GoogleAuth()
        self.__gauth.LocalWebserverAuth()        # Creates local webserver and auto handles authentication
        self.__drive = GoogleDrive(self.__gauth) # Create GoogleDrive instance with authenticated GoogleAuth instance


    def __find_folders(self, fldname):
        file_list = self.__drive.ListFile({
            'q': "title='{}' and mimeType contains 'application/vnd.google-apps.folder' and trashed=false".format(fldname)
            }).GetList()
        return file_list


    def __upload_files_to_folder(self, fnames, folder):
        for fname in fnames: 
            nfile = self.__drive.CreateFile({'title':os.path.basename(fname),
                                    'parents':[{u'id': folder['id']}]})
            nfile.SetContentFile(fname)
            nfile.Upload() 