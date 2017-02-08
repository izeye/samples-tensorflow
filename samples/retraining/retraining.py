import numpy as np
import tensorflow as tf

imagePath = 'roses/10090824183_d02c613f10_m.jpg'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        del(graph_def.node[1].attr["dct_method"])
        tf.import_graph_def(graph_def, name='')

def run_inference_on_image():
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return None

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    create_graph()

    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
        predictions = session.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        print('Predictions: %s' % predictions)

        squeezed = np.squeeze(predictions)
        print('Squeezed predictions: %s' % squeezed)

        argsorted = squeezed.argsort()
        print('Sorted predictions: %s' % argsorted)

        # Extract last 5 elements and reverse them.
        top_k = argsorted[-5:][::-1]

        f = open(labelsFullPath, 'rb')
        lines = f.readlines();
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = squeezed[node_id]
            print('%s (score = %.5f)' % (human_string, score))
        return labels[top_k[0]]

if __name__ == '__main__':
    answer = run_inference_on_image()
    print(answer)
