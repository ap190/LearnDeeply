import os
import json
import datetime
from util.settings import Settings
from .util import check_folder


class Datasaver:
    def save_profile_json(username, information):
        check_folder(Settings.profile_location)
        if (Settings.profile_file_with_timestamp):
            file_profile = os.path.join(Settings.profile_location, username + '_' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H-%M-%S") + '.json')
        else:
            file_profile = os.path.join(Settings.profile_location, username + '.json')

        with open(file_profile, 'w') as fp:
            fp.write(json.dumps(information, indent=4))

    def save_profile_commenters_txt(username, user_commented_list):
        pass
