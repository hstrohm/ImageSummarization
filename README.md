## Image Summarization

### Authors: Hanna Strohm, Nathan White, Joseph Mohr 

[Project Proposal](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Proposal.pdf)

[Midterm Report](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Midterm%20Report.pdf)

(these sections are all copied from various places as a starting point, and all should be edited)

### Introduction
For our project we decided to preform image summaration, which we define as the process of automatically describing images. Verbally describing images and their contents is an easy task for humans, but not necessarily so for computers. Both object detection and recognition and the relationships of different objects within a scene can be complex when it comes to image processing. Machine Learning has provided a viable solution to the first portion of this task, but the second portion remains difficult. We want to design an algorithmic approach that will be able to summarize the contents of a given image. This includes recognizing and describing common objects and their relation to other objects within the scene. After this, details regarding the objects of focus will also be added to the description. These two points will be the basis for the scene summarization with the main focus being the objects and how they relate to others within the scene, while extraneous details such as color and size can be added later. One example of this is an image of a human on their cell phone. While it’s easy to detect both objects individually, we want to be able to correlate the human with the phone to mean that “the human is using/holding/looking at the phone.” 

![image](https://user-images.githubusercontent.com/54555630/80897965-b87e0780-8cc3-11ea-8cab-7a5f6ae948bd.png) <br/>
Image Example (resize before) "The bear is black." is the computer generated response.

### Project Impact
The solution to this problem would automate the process of detecting objects, using that information to interpret the relationships between those objects, and finally translate that information into audio. There are a couple of reasons to solve this problem. First, it could provide access to images for people who are visually impaired or blind. The ability for a computer to analyze and describe an image accurately and in a way that is helpful would have the possibility of improving their daily experience navigating the world of images that exists today. Another useful application of image summarization would be to be able to include descriptions of images in an audio book, paper, or other text that is being listened to rather than read. Images can contain information important to a text, and therefore it would be useful to be able to include images in a text-to-speech algorithm. 

### Current State of the Art
Several studies have been conducted in regards to spatial based object association [1,2]. Falomir et al. [1] explored explaining images through an object’s visual characteristics as well as its spatial characteristics. All descriptions are qualitative, and not sentences meant for simply describing the image to the end-user. They also utilized a very small dataset, consisting only of a hallway from their building and the objects within it. Elliot et al. [2] explored the use of a visual dependency representation (VDR) model that looks at recognized object positions in order to determine how sentence structure should be formulated. For their study, they used a pre-trained R-CNN that would detect the objects for them, and utilized the detected objects and their image properties to describe the image’s content.
Another approach to this program has utilized a dependency tree recurrent neural network (DT-RNN) in order to associate text with an image [3]. The authors’ approach was to be able to map sentences to an image visualization of the text, and vice versa. They did not attempt to generate sentences for the given image, but rather used the sentences provided from their dataset. Lastly, even similarity based search has been explored [4]. Gathering images from Flickr, with their user specified captions, the authors simply search the dataset and attempt to find the image that matches the query the most. Once found, it takes the associated user-specified caption for the dataset image and assigns it to the query image.

### Our Process
#### Overview
In order to achieve our goal, we designed a system that would allow us to provide some textual descriptions of an input image. After supplying an input image, we perform multiple-object detection, getting boundary boxes, masks, and labels for each object. From here, the image is sent to two stages, a backdrop removal as well as an object property detector.
The backdrop removal process attempts to remove items that aren’t the focus of the image.
Using the remaining portions of the image, this is sent through a spatial relations detector. Which determines how two objects in an image are related in terms of their position in the image. From the results of the spatial relation detector, the objects and their relationships are turned into sentences using NLP techniques. For the object properties functions, these were intended to include object color, shape, etc; but due to time constraints only object color was implemented. The results of the object property functions are generated into simple sentences and then combined with the results of our NLP outputs to produce a textual description of the input image.

#### Object Detection
For object detection, we utilized a pretrained MASK RCNN from Python’s Pytorch package.
The images that were utilized for our project came from the COCO validation dataset. 
Passing the image through the model, it supplies us with bounding boxes, labels, and masks for each detected object.
This information is stored for later use in our process.

#### Backdrop Removal
Next the image is passed on to our backdrop removal process.
A natural first thought would be to use the masks to remove all non-detected objects.
However this doesn’t work, as your image may have extra detected objects that aren’t relevant to the description of the image.
For example, this picture of a tennis player keeps the detected people in the background when using the masks, just ignore the color skewing.
However, when using a contour based approach, which we will talk about in a moment, we get a better result.
With that in mind, we first tried an image segmentation tactic, hoping to cleanly remove the background while maining the main objects within the image.
However, we ended up with an approach that wasn’t able to do either.
Next, we moved on to a contour based approach. 
While this worked considerably better, you can see it still isn’t perfect.
However, due to time constraints we had to move on.
As a component of our project, we decided to narrow down the results of the object recognition down to the foreground and focused items within an image. This was decided because we can get many images that have lots of things happening in the background, but aren’t the actual focus of the image itself. However, being able to determine which items are the focus (or the main components) of the image has been difficult. We’re using OpenCV’s GrabCut which allows us to remove specific components of an image. We’ve attempted detecting foreground vs background objects by use of the contours within an image. 
However, this hasn’t given the desired results, as many non-focused and non-foreground items remain in the resulting image. One potential solution to explore would be to utilize the focus/blur detection techniques discussed in lecture. We could transform our input image using the Fast Fourier Transform and a Laplacian Kernel, then analyze the areas of higher frequency (ignoring the lower frequency parts of the image). This would give us the focused objects, but will require some experimentation with threshold values; as setting the threshold too high might remove some of the image context, while too low will include too much non-focused content. 

#### Spacial Relationships
After the backdrop is removed and the objects have been curated, we are left with just a few objects to determine how they relate to each other.
This gets sent through our simple spatial relationship detector, which uses the object bounding boxes to determine possible words or phrases that describe their relationship.
On screen, you can see that in the example of the dog and bowl, our detector returned a list of possible connecting phrases, which will be sent through our NLP algorithms to determine the best fitting connection.
#### Object Details
Due to time constraints, we were only able to implement a color detection algorithm. Originally we have plans to implement algorithms for determing object size and shape as well, but these didn't pan out. For determining object color, we utilized the original image and the bounding boxes of all the detected objects. For each object, we searched the bounding box to determine the most common color. Because objects have shading, it would be difficult to naively count the occurances of each RGB color present, much less assign a color label. Instead we used a k-means clustering algorithm with 5 clusters. We would iterate over every color in the image, assigning them to a cluster, and updating the cluster means. When the means stopped moving, we took the cluster mean (an RGB value) that had the most support (the most colors assigned to that cluster) to represent the object's color. This mean cluster RGB value was then compared to every pre-defined webcolor (from the Python webcolor package). For each comparison a simple distance metric was calculated using the following equation:
<img src="https://render.githubusercontent.com/render/math?math=\sum_{i \in {red, green, blue}} (i_{webcolor} - i_{cluster})^2">. The webcolor with the smallest distance value was then selected and its label was used to describe the object.

Below are results of our k-means approach for the image of a dog with a bowl. Please note that the graphical color representation of the clusters is not the color that the cluster itself is representing.

Dog with a bowl image:

![image](https://user-images.githubusercontent.com/35882267/80923181-91691980-8d47-11ea-901d-5a397f3453c5.png)


Results of k-means clustering for the dog object. Majority Cluster Color: Black

![image](https://user-images.githubusercontent.com/54555630/80898638-6345f400-8ccb-11ea-82c6-d19b806bc03f.png) 


Results of k-means clustering for the bowl object. Majority Cluster Color: Dim Grey

![image](https://user-images.githubusercontent.com/54555630/80898648-81abef80-8ccb-11ea-97fe-749e4b716a2f.png)


#### NLP: Sentence Building

### Evaluation
To evaluate our algorithm, we propose multiple methods. The first method would be to test if humans can determine which audio recording or text transcriptions match with which image. This will help us determine if our description is understandable and accurate. The next method would be to test if a computer algorithm can match our description with the correct image using the algorithm created by Socher et al. or [3] Finally, we can create a set of descriptions for each image by hand, and present them to people to see if they can pick out the generated description. This will help us to determine how natural our descriptions sound.

### Results

### Original Plan
We have not deviated too far from the original plan so far. The first step was to detect what the most prominent objects in the image were, where they were located, and other features about these objects such as color or if two objects are touching. We accomplished most of this using a pre-trained Mask R-CNN model that was trained on the COCO dataset, besides the extra features. We are currently working on determining relationships between objects using distance and NLP algorithms, which are a part of our original plan. The final step in our original plan was to use NLP to create actual sentences using word similarity. The following section has more details on what we have accomplished so far, and our updated plans for the future. The section following that details what didn’t work for us and what we are currently having issues with. <br/>
![image](https://user-images.githubusercontent.com/54555630/80898352-01d05600-8cc8-11ea-8208-25b5ccd6bdd3.png)

### Current Progress/New Updates
From our original plans, we have completed both object and boundary detections. We’re using a pre-trained Mask R-CNN model that was trained on the COCO dataset. This output of the model provides object labels and finds the boundaries of the detected object. From the recognized objects, we have a confidence threshold value of 75% that must be met for us to move forward with that object. This ignores most of the non-focal objects that were “detected.” The image above shows an example of an output image after this process.
At this point, we have a list of recognized objects and their bounding boxes. Our plan is to use the bounding boxes to generate a list of potential prepositions based on how the bounding boxes overlap. From this list of prepositions, combined with the labeled objects, we will use models from NLP in order to find the most likely way the two objects relate. For example, if we have a book above a table, possible prepositions include ‘on’, ‘in’, and ‘above’. The NLP model will inform us which potential phrasing is the most natural. To continue our previous example, our NLP model would tell us that “the book is on the table” is more natural than “the table is on the book.”
We plan to also use a similar NLP-based approach in order to determine the most likely action to be occurring between two objects. While unsure of its potential success, we hope that this will produce moderate results. 
Depending on how the above goes, we may have to cut out our plan to add object details. However, we have a plan to get basic information about items from the image. This would likely just add up to two adjectives per sentence, which would help with realism. If rushed, though, it could introduce inaccuracies. 


We expect to show examples of images and their generated summaries. We can also show what the algorithm produced at different stages of development. We will also outline and analyze the results of our evaluation. 

### References
1. Falomir, Zoe, et al. "Describing images using qualitative models and description logics." Spatial Cognition & Computation 11.1 (2011): 45-74.
2. Elliott, Desmond, and Arjen de Vries. "Describing images using inferred visual dependency representations." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2015.
3. Socher, Richard, et al. "Grounded compositional semantics for finding and describing images with sentences." Transactions of the Association for Computational Linguistics 2 (2014): 207-218.
4. Ordonez, Vicente, Girish Kulkarni, and Tamara L. Berg. "Im2text: Describing images using 1 million captioned photographs." Advances in neural information processing systems. 2011.

### Files:
**Code:**
* grab_model.ipynb - old notebook...think it was just testing the COCO dataset
* maskrcnn.ipynb - does vision stuff - gets data to be used by nlpstuff to make sentences
* nlpstuff.ipynb - does NLP stuff - makes sentences

**JSON:**
* results.json, failed.json - old method to transfer data about one image to NLP notebook
* everything.json - transfers data about (currently) 200 images to NLP notebook
* final_short_results.json - final results from NLP notebook, shortened version
* final_long_results.json - unshortened version
