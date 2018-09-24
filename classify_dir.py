# my classifier

from inception3_classifier import tf, run_inference_on_image, maybe_download_and_extract
import os, sys

image_dir = './images'


def main(_):
    maybe_download_and_extract()
    dlist = os.listdir(image_dir)
    idx = 0
    for f in dlist:
        if 'jpg' in f.lower():
            data = {}
            idx += 1
            # attempt classification
            data = run_inference_on_image(os.path.join(image_dir, f), num_top_predictions=1, verbose=False)
            print("{}. file: {} score: {:4.3f} id: {}".format(idx, f, data[0].get('score'), data[0].get('ident') ))
    print("done")
    sys.exit(0)
            
if __name__ == '__main__':
    tf.app.run(main=main)
        