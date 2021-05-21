import time
#import pyttsx3
import speech_recognition as sr
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

def speak_now(speech_text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 125)
    #print(engine.getProperty('rate'))
    engine.say(speech_text)
    engine.runAndWait()

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ######################
        image_h, image_w, _ = frame.shape
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        classes=['Scalpel','Straight_dissection_clamp','Straight_mayo_scissor','Curved_mayo_scissor']
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > 4: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            fontScale = 0.5
            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            if(X==class_ind and score> 85):
                print(coor[0],coor[1],coor[2],coor[3])
                print(classes[class_ind])
                print(score,end="%")
                print("")
                print("#########")
        #########################

        if not FLAGS.dont_show:
            cv2.imshow("result", result)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

# def speech():
#     r2 = sr.Recognizer()
#     r3 = sr.Recognizer()

#     #times = 100
#     while True:
#         with sr.Microphone() as source :
#             print("Speak Now")
#             #speak_now("Speak Now")
#             r3.adjust_for_ambient_noise(source)
#             audio = r3.listen(source)
#             try:
#                 if 'hello' in r3.recognize_google(audio):
#                     with sr.Microphone() as source :
#                         print("Search your query")
#                         #speak_now("Search your query")
#                         r2.adjust_for_ambient_noise(source)
#                         audio = r2.listen(source)

#                         try:
#                             get = r2.recognize_google(audio)
#                             print(get)
#                         except sr.RequestError as e:
#                             print('Failed'.format(e));
#                 elif 'exit' in r3.recognize_google(audio):
#                     print('Exit')
#                     #speak_now("Buh bye!")
#                     break
#                 else:
#                     continue
#             except:
#                 continue


r3 = sr.Recognizer()
r2 = sr.Recognizer()
#speech()
def speech():

    print("Search your query")
    #speak_now("Search your query")
    while True:
        with sr.Microphone() as source :
            r2.adjust_for_ambient_noise(source)
            audio = r2.listen(source,timeout =5)
            try:
                get = r2.recognize_google(audio,language = "en-GB")
                #print(get)
                x=get.lower()
                return x
            except sr.RequestError as e:
                print('Failed'.format(e))

            except sr.WaitTimeoutError:
                print('timeout')
            except:
                print("Please Repeat")
                #speak_now("please repeat")

dict={"scalpel":0,"pal pal":0,"clamp":1,"scissors":2,"scissor":2}
X=0

if __name__ == '__main__':
    try:
        while True:
            with sr.Microphone() as source :
                r3.adjust_for_ambient_noise(source)
                audio = r3.listen(source)
                try:
                    if 'hello' in r3.recognize_google(audio):
                        x=speech()

                        print(x)
                        if x in dict.keys():
                            X=dict[x]
                            print(str(X))
                            #image()
                            app.run(main)
                    elif 'exit' in r3.recognize_google(audio):
                        print('Exit')
                        #speak_now("Good bye")
                        break
                except:
                    continue


    except SystemExit:
        pass
