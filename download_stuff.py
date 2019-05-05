import os
import shutil
import zipfile
import tools
import sys
# from google_drive_downloader import GoogleDriveDownloader as gdd

if not os.path.isdir(os.path.join(tools.get_base_dir(), 'google-drive-downloader')):
    os.system('git clone https://github.com/ndrplz/google-drive-downloader.git')
sys.path.append(os.path.join(tools.get_base_dir(), 'google-drive-downloader'))
from google_drive_downloader.google_drive_downloader import GoogleDriveDownloader as gdd

url_weights = 'https://drive.google.com/open?id=1E7n8oOvP0Z2wXqGnm08BQGonUlvWB-VH'
weights_path_download = os.path.join(tools.get_base_dir(), 'vgg_16_for_ssd.zip')
weights_path_dst = os.path.join(tools.get_base_dir(), 'weights', 'vgg_16_for_ssd')

if os.path.isdir(weights_path_dst):
    print('It seems there weights already exists.')
else:
    # print('Downloading weights...')
    # print('wget ' + url_weights)
    # os.system('wget ' + url_weights)
    # if os.path.exists(weights_path):
    #     print('Weights downloaded successfully!')
    # else:
    #     print('Cannot find downloaded weights. They should be at ' + weights_path)
    #     quit()

    print('Downloading weights...')
    weights_dir = os.path.join(tools.get_base_dir(), 'weights')
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    file_id = '1E7n8oOvP0Z2wXqGnm08BQGonUlvWB-VH'
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=weights_path_dst,
                                        unzip=True)

    # print('Moving weights...')
    # weights_dir = os.path.join(tools.get_base_dir(), 'weights')
    # if not os.path.isdir(weights_dir):
    #     os.makedirs(weights_dir)
    # shutil.move(weigths_path, weigths_path_dst)
    # print('Weights moved.')

    print('Unziping weights...')
    weights_dir = os.path.join(tools.get_base_dir(), 'weights')
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    zip_ref = zipfile.ZipFile(weights_path, 'r')
    zip_ref.extractall(weights_path_dst)
    zip_ref.close()
    print('Weights unziped.')

# url_voc0712 = 'https://drive.google.com/open?id=1TtO5FD2g2bmzfyyGsqp7fnNvZi_0Qp-2'
# voc0712_filename = 'VOC0712.rar'
# voc0712_path = os.path.join(tools.get_base_dir(), voc0712_filename)
# voc0712_path_dst = os.path.join(tools.get_base_dir(), 'datasets', voc0712_filename)
#
# print('Downloading VOC0712...')
# os.system('wget ' + url_voc0712)
# if os.path.exists(voc0712_path):
#     print('VOC0712 downloaded successfully!')
# else:
#     print('Cannot find downloaded VOC0712. It should be at ' + voc0712_path)
#     quit()
#
# print('Moving VOC0712 dataset...')
# datasets_dir = os.path.join(tools.get_base_dir(), 'datasets')
# if not os.path.isdir(datasets_dir):
#     os.makedirs(datasets_dir)
# shutil.move(voc0712_path, voc0712_path_dst)
# print('VOC0712 moved.')




