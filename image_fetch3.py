# requires Python3
"""
image_fetch.py

INTENT: this program will download a set of image files and name the files with the
captureEventID, species, count

based on "standard" data products from Snapshot Serengeti Project
1) all_images.csv (supplied by Snapshot Serengeti team)
2) gold_standard_data.csv (supplied by T.M. Anderson)
3) consensus_data.csv (a huge CSV file containing all the data for every captureEventID)

Consensus data product is ever-growing larger!  So this will change over time.
Gold Standard Data comes from the Wyoming group (Clune's lab)

e.g. ASG0bx6j_jackal_1.jpg (would represent capture event ASG0bx6j, consensus species jackal,
species count of 1)

 USAGE:

    python image_fetch <capture_id_to_download_csv_file> [options]

    options:
    --base_url <url_of_image_server>
               (note: default is UM's server)

    --dest_folder <folder_to_receive_downloads>
               (note: default= './downloads')

    --image_reference_file <file_of_capture_id_and_path_locations>
                (note: default='./all_images.csv')

capture_events.csv - a CSV formatted file where the first column contains the CaptureEventID you would like to download
base_url is https://snapshotserengeti.s3.msi.umn.edu/ (we don't expect this to change)
dest_folder by default is ./downloads, this can be altered by your command line parameter
"""
BASE_URL = "https://snapshotserengeti.s3.msi.umn.edu/"
IMAGE_REFERENCE_FILE = './all_images.csv'
DEST_FOLDER = './downloads'

# Typical data products used
# all_images.csv - has two columns, one is the consensus ID, the other a pathname of the picture (relative to the root of the server)
# gold_standard_data.csv (from Wyo. paper)
# consensus_data.csv - all the consensus data (best suited for a pandas data frame)

import csv
import os
import sys
from urllib.request import urlopen


def download_picture(url, dest_filename, dest_folder='./', overwrite=False, verbose=False):
    """download_picture(url, dest_filename, dest_folder, overwrite=False)
       note: dest_folder='./', overwrite=False, verbose=False
       if overwrite is True, it will overwrite an existing file
       if verbose, show user some information on console about progress.

       This is a synchronous process, so that could be problematic
       when image server is not responding properly

       Should trap errors when downloading or saving the file
    """
    try:
        response = urlopen(url)
        document = response.read()
    except Exception as e:
        print("Encountered an error on {}".format(url))
        print(str(e))
        return False

    outfile = os.path.join(dest_folder, dest_filename)
    if not os.path.isdir(dest_folder):
                # make directory if required
        os.mkdir(dest_folder)

    if os.path.isfile(outfile) and overwrite == False:
        if verbose: print("skipping overwrite of {}".format(outfile))
        return False # don't overwrite existing file
    else:
        if verbose: print("saving file {}".format(outfile))

        try:
            with open(outfile,'wb') as fp:
                fp.write(document)
            return True
        except Exception as e:
            print("Encountered an error saving {}".format(outfile))
            print(str(e))
            return False

def urlfilename(url):
    """return the trailing filename from a url"""
    parts = url.split('/')
    return parts[len(parts)-1]


def test1():
    """download a single picture from a URL"""
    test_url = "https://snapshotserengeti.s3.msi.umn.edu/S1/E06/E06_R1/S1_E06_R1_PICT0009.JPG"
    result = download_picture(test_url, dest_filename="S1_E06_R1_PICT0009.JPG", dest_folder='./images', overwrite=True)
    if result:
        print("success downloading {}".format(urlfilename(test_url)))

def get_image_data(csv_file_name, base_url, image_reference_file='./all_images.csv'):
    """get_image_urls - returns all the urls associated with the capture event ids in the csv file
    the list is composed of tuples, (capture_event_id, count, species, url)
    """
    urls = []
    all_images_file = image_reference_file
    # treat the images file as a HUGE text buffer
    # this will speed up access
    with open(all_images_file, 'r') as fp:
        data = fp.read()
        
    with open(csv_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                # header row
                continue
            
            # find location of the index in the buffer
            idx = data.find(row[0])
            if idx > 0:
                # find end of line index
                eol = data.find('\n', idx)
                # path is located in second column
                path = data[idx:eol-1].split(',')[1]
                path = path.replace("\"","") # clean up extraneous double-quotes
                # make a tuple of capture_id, species, count, and its real_world URL location
                urls.append( (row[0], row[1], row[2], os.path.join(base_url, path)) )
                if line_count % 100 == 0:
                    # show user some progress to keep them convinced we aren't hung up
                    print("processed {} images".format(line_count))
    return urls
            
    
def usage():
    """indicate usage arguments to the console and exit"""
    usage_str = """
    USAGE:

    python image_fetch <capture_id_to_download_csv_file> [options]

    options:
    --base_url <url_of_image_server>
               (note: default is UM's server)

    --dest_folder <folder_to_receive_downloads>
               (note: default= './downloads')

    --image_reference_file <file_of_capture_id_and_path_locations>
                (note: default='./all_images.csv')
    """
    print(usage_str)
    sys.exit(0) # clean exit to OS

def main(args):
    """main image download handling routine"""
    csv_file_name = None
    
    
    for i in range(0,len(args)):
        if '.csv' in args[i].lower():
            csv_file_name = args[i]
            
        if '--base_url' in args[i]:
            # override base_url (by default at UNM)
            BASE_URL = args[i+1]
            
        if '--dest_folder' in args[i]:
            # overide of default destination folder
            DEST_FOLDER = args[i+1]

        if '--image_reference_file' in args[i]:
            # override of default image reference file "./all_images.csv"
            IMAGE_REFERENCE_FILE = args[i+1]

            
    if csv_file_name is None:
        print("Missing CSV file")
        usage()
        raise ValueError
    
    # get all image data, assumes the data is small enough
    print("generating image data")
    data = get_image_data(csv_file_name, BASE_URL, IMAGE_REFERENCE_FILE)
    print("downloading pictures--")
    for item in data:
        # item will contain: [captureEventID, species, count, url]
        url = item[3]
        dest_filename = '{}_{}_{}.jpg'.format(item[0], item[2], item[1])
        print("attempting to download {}".format(url))
        try:
            download_picture(url, dest_filename=dest_filename, dest_folder=DEST_FOLDER, overwrite=False)
            print(dest_filename + ", success")
        except Exception as e:
            print("Error downloading {} with error {}".format(url, str(e)))
                  
    print('Done.')
            

if __name__ == '__main__':
    main(sys.argv)
