import cv2
import numpy as np
from model_ML import create_model_pretrain
import time
from data_helper import calculateRGBdiff
import imageio
###### Parameters setting
dim = (224,224) # MobileNetV2
n_sequence = 8
n_channels = 3 # RGB Channel
n_output = 18 # number of output class
weights_path = 'weight-513-0.80-0.89.hdf5' # pretrain weight path
#######


### load model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output)
model.load_weights(weights_path)

### Define empty sliding window
frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

### State Machine Define
RUN_STATE = 0
WAIT_STATE = 1
SET_NEW_ACTION_STATE = 2
state = RUN_STATE # 
previous_action = -1 # no action
text_show = 'no action'

### Class label define
class_text = [
'1 Horizontal arm wave',
'2 High arm wave',
'3 Two hand wave',
'4 Catch Cap',
'5 High throw',
'6 Draw X',
'7 Draw Tick',
'8 Toss Paper',
'9 Forward Kick',
'10 Side Kick',
'11 Take Umbrella',
'12 Bend',
'13 Hand Clap',
'14 Walk',
'15 Phone Call',
'16 Drink',
'17 Sit down',
'18 Stand up']

video_path = r'/content/gdrive/MyDrive/CSE 598/Testing /TestVideo1'
cap = cv2.VideoCapture(video_path)
start_time = time.time()
output_frame = []
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()  
    count = count +1
    
    if ret == True:
        
        new_f = cv2.resize(frame, dim)
        new_f = new_f/255.0
        new_f_rs = np.reshape(new_f, (1, *new_f.shape))
        frame_window = np.append(frame_window, new_f_rs, axis=0)
        

        ### if sliding window is full(8 frames), start action recognition
        if frame_window.shape[0] >= n_sequence:
            frame_window_dif = calculateRGBdiff(frame_window.copy())
            frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)
            # print(frame_window_new.dtype)
            ### Predict action from model
            output = model.predict(frame_window_new)[0]           
            predict_ind = np.argmax(output)
            
            ### Check noise of action
            if output[predict_ind] < 0.55:
                new_action = -1 # no action(noise)
            else:
                new_action = predict_ind # action detect

            ### Use State Machine to delete noise between action(just for stability)
            ### RUN_STATE: normal state, change to wait state when action is changed
            if state == RUN_STATE:
                if new_action != previous_action: # action change
                    state = WAIT_STATE
                    start_time = time.time()     
                else:
                    if previous_action == -1:
                        text_show = 'no action'                                              
                    else:
                        text_show = "{: <22}  {:.2f} ".format(class_text[previous_action],
                                    output[previous_action] )
                    print(text_show)  

            ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
            elif state == WAIT_STATE:
                dif_time = time.time() - start_time
                if dif_time > 0.5: # wait 0.5 second
                    state = RUN_STATE
                    previous_action = new_action

            ### put text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text_show, (10,450), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)   
            
            ### shift sliding window
            frame_window = frame_window[1:n_sequence]
            
            ### To show dif RGB image
            # vis = np.concatenate((new_f, frame_window_new[0,n_sequence-1]), axis=0)
            # cv2.imshow('Frame', vis)
            #cv2.imshow('Frame', frame)
           # cv2.imwrite(r'/content/gdrive/MyDrive/CSE 598/Testing /temp/'+str(count)+'.jpg',frame)
            output_frame.append(frame)
            

        ### To show FPS
        # end_time = time.time()
        # diff_time =end_time - start_time
        # print("FPS:",1/diff_time)
        # start_time = end_time
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 
    else: 
        break
print(type(output_frame))
output_path = r'output1.mp4'
fps = 0.5
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MP4V'), 20, dim)
# 0x00000021
output_size = output_frame[0].shape[1], output_frame[0].shape[0]
out = cv2.VideoWriter(output_path,fourcc, 20, output_size)
print(len(output_frame))
for i in range(len(output_frame)):
    # writing to a image array
    out.write(output_frame[i])
out.release()

cap.release()

