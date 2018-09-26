# my classifier

from inception3_classifier import tf, run_inference_on_image, maybe_download_and_extract
import os, sys, csv

from tinymongo import TinyMongoClient
db = TinyMongoClient().image_database

image_dir = './downloads'
output_file = 'gs_classification.csv'

def build_database(fname):
    """build a tinymongo database for classifications from existing CSV"""
    try:
        with open(fname, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                item = db.classified.find_one({'captureEventID':row.get('captureEventID')})
                if item is None:
                    db.classified.insert_one(row)  
    except:
        data = None
    
    

def get_previous_classifications(fname):
    """get all previously classified data"""
    try:
        with open(fname, 'r') as csvfile:
            data = csvfile.read()
    except:
        data = ''
    return data

def main(_):
    maybe_download_and_extract()
    # build_database(output_file)
    # classified_data = get_previous_classifications(output_file)
    dlist = os.listdir(image_dir)
    idx = 0
    for f in dlist:
        if 'jpg' in f.lower():
            data = {}
            idx += 1
            # isolate consensus data
            consensus = f.split('_')
            # check if it has been classified
            item = db.classified.find_one({'captureEventID':consensus[0]})
                
            if item:
                print("skipping {}".format(consensus[0]))
            else:
                data = run_inference_on_image(os.path.join(image_dir, f), num_top_predictions=1, verbose=False)
                print("{}. file: {} score: {:4.3f} id: {}".format(idx, f, data[0].get('score', None), data[0].get('ident') ))
                item = {'captureEventID':consensus[0], 'species':consensus[1],
                        'score': str(data[0].get('score', '')), 'ident': data[0].get('ident','')}
                sdata = '{},{},{},\"{}\"\n'.format(consensus[0], consensus[1], data[0].get('score',''), data[0].get('ident',''))
                db.classified.insert_one(item)
                
                with open(output_file, 'a') as fh:
                    fh.write(sdata)
        
    print("done")
    sys.exit(0)
            
if __name__ == '__main__':
    tf.app.run(main=main)
