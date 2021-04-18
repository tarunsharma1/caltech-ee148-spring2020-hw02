import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    heatmap = np.random.random((n_rows, n_cols))


    # normalize image
    I = I/255
    t_rows, t_cols, _ = np.shape(T)
    
    template_patch_flattened_norm = np.sqrt(np.sum(T.flatten()**2))

    # padding in order to maintain the same shape for heatmap and image
    # the formula for calculating the resultant shape after colvolution is -> rows_result = (image_rows + padding_rows - template_rows)/stride + 1
    # in this case we want rows_result = image_rows, and same for cols, hence we get 
    padding_rows = (n_rows - 1)*stride + t_rows - n_rows
    padding_cols = (n_cols - 1)*stride + t_cols - n_cols


    # the resultant matrix of the convolution will be smaller than the original image, hence zero padding it around so that the resultant is the same size
    # as original image
    padded_image = np.zeros((padding_rows+n_rows, padding_cols+n_cols,n_channels))
    # insert original image in the middle of this matrix
    
    padded_image[int(padding_rows/2):int(padding_rows/2)+n_rows, int(padding_cols/2):int(padding_cols/2)+n_cols, :] = I

    padded_n_rows, padded_n_cols, _ = np.shape(padded_image)
    # I think window size can be my scaling factor thing i.e window size in the image

    for i_h,i in enumerate(range(0, padded_n_rows - t_rows + 1, stride)):
        for j_h,j in enumerate(range(0, padded_n_cols - t_cols + 1, stride)):
            
            window = padded_image[i:i+t_rows, j:j+t_cols, :]
            heatmap[i_h,j_h] = np.sum(np.multiply(window, T))

            # divide that by the norms of the template and the window in order to get the cosine distance (matched filtering) 
            window_flattened_norm = np.sqrt(np.sum(window.flatten()**2))
            
            heatmap[i_h,j_h] = heatmap[i_h,j_h]/(window_flattened_norm)            
    

    heatmap = heatmap/template_patch_flattened_norm
    # threshold the output of the convolution here in order to reduce bounding boxes produced...the averaged heatmap will have our confidence values
    heatmap[heatmap < 0.9] = 0

    # round off heatmap to save space
    #heatmap = np.round(heatmap, 2)
            

    '''
    END YOUR CODE
    '''
    
    #Image.fromarray(np.array(heatmap*255,dtype='uint8')).show()

    return heatmap


def predict_boxes(heatmap,I,t_rows, t_cols,stride=1):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    (n_rows,n_cols,n_channels) = np.shape(I)

    padding_rows = (n_rows - 1)*stride + t_rows - n_rows
    padding_cols = (n_cols - 1)*stride + t_cols - n_cols
    
    threshold = 0.1
    x = np.where(heatmap > threshold)
    if x[0].shape[0] == 0:
        print ('no bounding boxes detected')
        return []

    for i in range(0,x[0].shape[0]):
        row,col = x[0][i], x[1][i]
        # now map back to values to the image that we convolved on - padded image..assuming stride is 1
        tl_row = int(stride*row)
        tl_col = int(stride*col)
        br_row = tl_row + int(t_rows)
        br_col = tl_col + int(t_cols)

        # these values are for the padded image..for the original image, we have to make sure that the rows
        # and cols are within the I.shape..if rectangle is in padded region then we ignore it
        if tl_row > int(padding_rows/2) and tl_row < (n_rows + int(padding_rows/2)):
            if tl_col > int(padding_cols/2) and tl_col < (n_cols + int(padding_cols/2)):
                output.append([int(tl_row - int(padding_rows/2)) ,int(tl_col - int(padding_cols/2)), int(br_row -int(padding_rows/2)), int(br_col-int(padding_cols/2)), round(float(heatmap[row,col]),2)])
                
    

    # im = Image.fromarray(I)
    # draw = ImageDraw.Draw(im)
    # for i in range(0,len(output)):
    #     pt1,pt2,pt3,pt4 = output[i][0], output[i][1], output[i][2], output[i][3]
    #     #153 316 171 324
    #     draw.rectangle([(pt2,pt1),(pt4,pt3)], outline="green")
        
    # im.show()

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    # we have 3 templates and we want to take the average of them
    heatmap_matrix = np.zeros((I.shape[0],I.shape[1], len(list_of_patches)))
    for i,T in enumerate(list_of_patches):

        heatmap = compute_convolution(I, T)
        heatmap_matrix[:,:,i] = heatmap
    
    # average over the heatmaps
    heatmap_avg = np.mean(heatmap_matrix, axis=-1)
    #import ipdb;ipdb.set_trace()
    t_rows, t_cols, _ = T.shape # all the templates are the same size
    output = predict_boxes(heatmap_avg,I,t_rows, t_cols)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/RedLights2011_Medium'

# load splits: 
split_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '/home/tarun/Downloads/caltech-ee148-hw-data/hw2/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True



# I am going to save templates from a specific image, 'RL-001.jpg', by manually
# cropping out three instances of red traffic lights. I am going to then rule this image out
# of the test set if it is present in the test set.

list_of_patches = []
I = Image.open(os.path.join(data_path,'RL-001.jpg'))
I = np.asarray(I)
I = I/255.0
# I manually found these values
im1 = I[154:172, 316:324,:]
im2 = I[192:205,419:427,:]
im3 = I[179:201,65:81,:]

#  resize all templates to one the size of im2 (largest)
im1 = Image.fromarray(np.array(im1*255,dtype='uint8')).resize((im2.shape[1],im2.shape[0]))
im1 = np.asarray(im1)/255.0
im3 = Image.fromarray(np.array(im3*255,dtype='uint8')).resize((im2.shape[1],im2.shape[0]))
im3 = np.asarray(im3)/255.0


# weakened version has only first patch
list_of_patches.append(im1)
list_of_patches.append(im2)
list_of_patches.append(im3)

if 'RL-001.jpg' in file_names_test:
    print ('removing image 1 from test set ')
    file_names_test.remove('RL-001.jpg')



'''
Make predictions on the training set.
'''
preds_train = {}

for i in range(len(file_names_train)):
#for i in range(0,20):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)


    preds_train[file_names_train[i]] = detect_red_light_mf(I)

#save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
