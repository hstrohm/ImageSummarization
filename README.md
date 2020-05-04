## Image Summarization

### Authors: Hanna Strohm, Nathan White, Joseph Mohr 

[Project Proposal](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Proposal.pdf)

[Midterm Report](https://github.com/hstrohm/ImageSummarization/blob/master/CS766%20Midterm%20Report.pdf)

(these sections are all copied from various places as a starting point, and all should be edited)

### Introduction
For our project we decided to preform image summaration, which we define as the process of automatically describing images. Verbally describing images and their contents is an easy task for humans, but not necessarily so for computers. Both object detection and recognition and the relationships of different objects within a scene can be complex when it comes to image processing. Machine Learning has provided a viable solution to the first portion of this task, but the second portion remains difficult. We want to design an algorithmic approach that will be able to summarize the contents of a given image. This includes recognizing and describing common objects and their relation to other objects within the scene. After this, details regarding the objects of focus will also be added to the description. These two points will be the basis for the scene summarization with the main focus being the objects and how they relate to others within the scene, while extraneous details such as color and size can be added later. One example of this is an image of a human on their cell phone. While it’s easy to detect both objects individually, we want to be able to correlate the human with the phone to mean that “the human is using/holding/looking at the phone.” 

Image Example: "The bear is black." is the computer generated response.|
:-------------------------:|
<img src="https://user-images.githubusercontent.com/54555630/80897965-b87e0780-8cc3-11ea-8cab-7a5f6ae948bd.png" width="400" /> |


### Project Impact
The solution to this problem would automate the process of detecting objects, using that information to interpret the relationships between those objects, and finally translate that information into audio. There are a couple of reasons to solve this problem. First, it could provide access to images for people who are visually impaired or blind. The ability for a computer to analyze and describe an image accurately and in a way that is helpful would have the possibility of improving their daily experience navigating the world of images that exists today. Another useful application of image summarization would be to be able to include descriptions of images in an audio book, paper, or other text that is being listened to rather than read. Images can contain information important to a text, and therefore it would be useful to be able to include images in a text-to-speech algorithm. 

### Current State of the Art
Several studies have been conducted in regards to spatial based object association [1,2]. Falomir et al. [1] explored explaining images through an object’s visual characteristics as well as its spatial characteristics. All descriptions are qualitative, and not sentences meant for simply describing the image to the end-user. They also utilized a very small dataset, consisting only of a hallway from their building and the objects within it. Elliot et al. [2] explored the use of a visual dependency representation (VDR) model that looks at recognized object positions in order to determine how sentence structure should be formulated. For their study, they used a pre-trained R-CNN that would detect the objects for them, and utilized the detected objects and their image properties to describe the image’s content.
Another approach to this program has utilized a dependency tree recurrent neural network (DT-RNN) in order to associate text with an image [3]. The authors’ approach was to be able to map sentences to an image visualization of the text, and vice versa. They did not attempt to generate sentences for the given image, but rather used the sentences provided from their dataset. Lastly, even similarity based search has been explored [4]. Gathering images from Flickr, with their user specified captions, the authors simply search the dataset and attempt to find the image that matches the query the most. Once found, it takes the associated user-specified caption for the dataset image and assigns it to the query image.

### Our Process
#### Overview
In order to achieve our goal, we designed a system that would allow us to provide some textual descriptions of an input image. Below is an pictural overview of how our system works. Starting from the left side, we begin with supplying the system with some input image. Next we perform multiple-object detection, getting boundary boxes, masks, and labels for each detected object in the image. From here, the image is sent to two stages, a backdrop removal as well as an object property detector. The backdrop removal process attempts to remove items that aren’t the focus of the image, as well as the background scene. During this phase, stored objects that are no longer in the image are removed. The remaining objects and image is then sent through a spatial relations detector, which determines how two objects in an image are related in terms of their relative position in the image. For the object properties functions, these were initially intended to include object color, shape, and size; but due to time constraints only object color was implemented. The results of the color detection are used to describe the objects within the NLP generated sentences. Using the results of both the spatial relation detector and object properties, our NLP algorithm generates a textual description of the image.

![image](https://user-images.githubusercontent.com/35882267/80925353-b912ae80-8d54-11ea-8250-aa6b477b6252.png)


#### Object Detection
For object detection, we utilized a pretrained MASK RCNN from Python’s Pytorch package, with images from the COCO validation dataset. Each image that is passed through the model results in bounding boxes, labels, and masks for each detected object in the image. This information is stored for later use in our process. The model detects many objects that either aren't there or are incorrectly detected. To counteract this, we set a confidence threshold of 75% that must be met before we accept the classification of an object. This also had the added bonus of ignoring most of the non-focal objects within the image. Below is an example of below and after this threshold was set.

Before Threshold           | After Threshold           |
:-------------------------:|:-------------------------:|
![before_threshold](https://user-images.githubusercontent.com/35882267/80971919-0dfd0600-8de3-11ea-8b8a-b1ec31490d6e.png) | ![after_threshold](https://user-images.githubusercontent.com/35882267/80971921-0e959c80-8de3-11ea-949c-2bdf60910f3e.png)

Below is an example of the masks that were produced, along with their associated label.

Oven                       | Microwave                 | Potted Plant              |
:-------------------------:|:-------------------------:|:-------------------------:|
![mask1](https://user-images.githubusercontent.com/35882267/80972491-c7f47200-8de3-11ea-8669-d739554d8bf4.png) | ![mask2](https://user-images.githubusercontent.com/35882267/80972490-c7f47200-8de3-11ea-8ded-52b6fbe2ea8d.png) | ![mask6](https://user-images.githubusercontent.com/35882267/80972493-c88d0880-8de3-11ea-9edb-f0d3441c5910.png)

Given that we utilized a pretrained model for multi-object detection, we ran into very few errors. The biggest issue we faced was with the confidence intervals on detected objects and multiple labelings for each object, but this was easily remedied with a confidence threshold.

#### Backdrop Removal
For the backdrop removal, we settled with a contour based approach. It worked reasonably well with most images, usually only failing to remove a few background objects. At first, we considered using a mask based approach to remove the background. This would work, but we would be left with all detected objects, something that wasn't desired, rather than the removal of most of them. Below is an example of this, using a picture of a man playing tennis with onlookers.


Oringal Image              | Contour Approach          | Mask Approach             
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/35882267/80924212-4c47e600-8d4d-11ea-8b70-6ecb1387235a.png" width="181" height="279" /> | <img src="https://user-images.githubusercontent.com/35882267/80924214-4ce07c80-8d4d-11ea-9db5-558fbfa572b0.png" width="181" height="279" /> | <img src="https://user-images.githubusercontent.com/35882267/80924215-4ce07c80-8d4d-11ea-98f3-f823b507c4bb.png" width="181" height="279" />


Ignoring the color issues in producing the mask based image, we can see that the contour approach performs better as many of the background objects are removed. While some background obejct still remain in this example, we are able to generate a more accurate description, as it won't discuss extraneous background people. However, in another example - pictured below - we can see the opposite happens, much of dog is removed from the image.

Original Image             | Contours                  | Result                   
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/35882267/80923897-e9098400-8d4b-11ea-84d3-1562082cb323.png) | ![image](https://user-images.githubusercontent.com/35882267/80924187-2e7a8100-8d4d-11ea-8535-6fcdd287f800.png) | ![image](https://user-images.githubusercontent.com/35882267/80924189-2e7a8100-8d4d-11ea-9316-2106579b865f.png)


As part of the backdrop removal process, we reduce the number of detected objects. Starting with all detected objects and the contoured image, we iterate over every object. If every point within the object's mask is removed (i.e. completely black), then the object's data is thrown out. After all removed objects are thrown out, we send the results to the spatial relations detector.


#### Backdrop Issues
As a component of our project, we decided to narrow down the results of the object recognition down to the foreground and focused items within an image. This was decided because we can get many images that have lots of events occuring in the background, but aren’t the actual focus of the image itself. However, being able to determine which items are the focus (or the main components) of the image has been difficult. We’re using OpenCV’s GrabCut which allows us to remove specific components of an image. We’ve attempted detecting foreground vs background objects by use of the contours within an image. 
However, this hasn’t given the desired results, as several non-focused and non-foreground items remain in the resulting image. We were unable to perfect this approach, and time ran out for us to do so.

We attempted an image segmentation tactic prior to learning about it in lecture, hoping it would cleanly remove the background and background objects. However, it performed very poorly. It removed inconsistent patches throughout the image, and wasn't very usable.

Original Image             | Segmentation Mask         | Segmented Image           
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/35882267/80923897-e9098400-8d4b-11ea-84d3-1562082cb323.png) | ![image](https://user-images.githubusercontent.com/35882267/80923872-ddb65880-8d4b-11ea-94ac-55270f7ef51f.png)  |  ![image](https://user-images.githubusercontent.com/35882267/80923874-dee78580-8d4b-11ea-8ef8-fcdca9148e4d.png)

We also attempted to explore the focus/blur detection techniques discussed in lecture. With the focus techniques, we were in a very similar position to our contour approach in some cases. But for many images, the main object in the foreground was removed, due to it being out of focus with the camera. Some hybrid approach of image focus, contours, and masking would probably work best for this project, however we were unable to implement a workable approach with our time constraints.

#### Spatial Relationships
After the backdrop is removed and the objects have been curated, we are left with just a few objects to determine how they relate to each other. Each pair of objects is sent through our spatial relationship detector. It uses the bounding boxes of both objects to determine how they relate spatially to each other. Below is an example of the dog and bowl image having been sent through our detector. The detector distinguishes between the primary and complementary objects. For the below example, dog was the primary object and the bowl is complementary.

Original Image             | List of connecting phrases
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/35882267/80923897-e9098400-8d4b-11ea-84d3-1562082cb323.png) | 1. is beside<br>2. is to the right of<br>3. is adjacent to<br>4. is holding<br>5. overlaps<br>

So, one possible sentence that could be generated from this image is "The dog is beside the bowl." From the bounding boxes of the objects, the spatial relations detector uses the corners to detect how the boxes relate spatially to each other. The first thing is checks, is whether the complementary box is fully within the primary box. This produces a list of connecting phrases such as "is within", "is in", "is on", "is behind" and "is in front of." Failing the first check, the dectector looks to see if the complementary box surround the primary box, in other words if the primary box is within the complementary box. This produces connecting phrases such as "surrounds", "encompasses", "is behind", and "is in front of." Then it checks to see if the two objects directly overlap each other, in the form of a t. This yeilds connections such as "overlaps", "is behind", and "is in front of." Lastly, it checks each of the cardinal directions (in the x,y coordinate space) to see if the complementary box is above, below, left, or right of the primary box, this includes partial overlapping (meaning the majority of the complementary box lies outside the primary box). This produces the remaining phrases such as "is beside", "is adjacent to", "is above", "is below", "is holding", "overlaps", etc.

#### Object Details
Due to time constraints, we were only able to implement a color detection algorithm. Originally we have plans to implement algorithms for determing object size and shape as well, but these didn't pan out. For determining object color, we utilized the original image and the bounding boxes of all the detected objects. For each object, we searched the bounding box to determine the most common color. Because objects have shading, it would be difficult to naively count the occurances of each RGB color present, much less assign a color label. Instead we used a k-means clustering algorithm with 5 clusters. We would iterate over every color in the image, assigning them to a cluster, and updating the cluster means. When the means stopped moving, we took the cluster mean (an RGB value) that had the most support (the most colors assigned to that cluster) to represent the object's color. This mean cluster RGB value was then compared to every pre-defined webcolor (from the Python webcolor package). For each comparison a simple distance metric was calculated using the following equation:
![image](https://user-images.githubusercontent.com/35882267/80968667-fa9b6c00-8ddd-11ea-8be0-1b428de963c3.png). The webcolor with the smallest distance value was then selected and its label was used to describe the object.

Below are results of our k-means approach for the image of a dog with a bowl. Please note that the graphical color representation of the clusters is not the color that the cluster itself is representing.

Dog with a bowl image:

![image](https://user-images.githubusercontent.com/35882267/80923181-91691980-8d47-11ea-901d-5a397f3453c5.png)


Results of k-means clustering for the dog object. Majority Cluster Color: Black

![image](https://user-images.githubusercontent.com/54555630/80898638-6345f400-8ccb-11ea-82c6-d19b806bc03f.png) 


Results of k-means clustering for the bowl object. Majority Cluster Color: Dim Grey

![image](https://user-images.githubusercontent.com/54555630/80898648-81abef80-8ccb-11ea-97fe-749e4b716a2f.png)


#### NLP: Sentence Building
(From Midterm report)
At this point, we have a list of recognized objects and their bounding boxes. Our plan is to use the bounding boxes to generate a list of potential prepositions based on how the bounding boxes overlap. From this list of prepositions, combined with the labeled objects, we will use models from NLP in order to find the most likely way the two objects relate. For example, if we have a book above a table, possible prepositions include ‘on’, ‘in’, and ‘above’. The NLP model will inform us which potential phrasing is the most natural. To continue our previous example, our NLP model would tell us that “the book is on the table” is more natural than “the table is on the book.”
We plan to also use a similar NLP-based approach in order to determine the most likely action to be occurring between two objects. While unsure of its potential success, we hope that this will produce moderate results. 
Depending on how the above goes, we may have to cut out our plan to add object details. However, we have a plan to get basic information about items from the image. This would likely just add up to two adjectives per sentence, which would help with realism. If rushed, though, it could introduce inaccuracies. 

### Evaluation and Results
We used a survey to gather information on how successful our captions were, which includes a matching section, and questions about the naturalness and helpfulness of the captions. The matching section had four questions that had a caption and three possible options for a corresponding image. For each question, the order of the options was randomized for each participant. The respondents got one question correct, and for the other three questions the correct answer was the second most common. This shows that the captions are not as effective as they could be yet. They are somewhat understandable but not completely clear. We also had the respondents rate how natural the captions sounded and there was an overall average of around 2.5, which is exactly in the middle of the rating scale that we used. The averages for the individual questions range from 1.6 to 3.6, which suggest that we could improve the naturalness of the captions quite a bit. It also suggests that some captions are better than others. We also measured helpfulness and found a similar range with an overall average 2.7. This suggests that the captions would need to improve the captioning before it could be useful to people. Overall, our captions were found to be alright but with a lot of room for improvement.

The captions that each of the next three tables reference are stored here.

| C1     | C2     | C3     | C4     |
|:------:|:------:|:------:|:------:|
| A darkslate grey bus is adjacent to a bus. A bus is below a silver bird. A bus is beside a bird. | The gray person is holding the black elephant. The person is holding the elephant. A elephant is beside a elephant. | A dark khaki person is to the right of a person. A person is to the right of a dark olive green banana. A person is above a banana. A person is above a banana. A banana is beside a banana. | A maroon teddy bear is holding a teddy bear. A teddy bear is holding a teddy bear. A teddy bear is holding a teddy bear. |

Below is an example of a matching question in the survey. Google Forms would not allow for the removal of the text: "Option 1", "Option 2", or "Option 3", as some text needed to be assigned to the image.

![image](https://user-images.githubusercontent.com/54555630/80942644-ed18be80-8daa-11ea-934a-9fbc0ca3c9ec.png) 

The following table and graph shows the number of people that chose each option for each question in the matching section. The correct answer for each is starred. As you can see, most people got caption four correct, but captions one, two, and three were not. However, for captions one, two, and three, the correct answer was the second most selected answer.

![Counts of Image Selection by Option](https://user-images.githubusercontent.com/35882267/80967266-ac856900-8ddb-11ea-91b9-14a4293d95ad.png)

|          | C1     | C2     | C3     | C4     |
|:--------:|:------:|:------:|:------:|:------:|
| Option 1 | 7*    	| 7*  	 | 8*	    | 18*    |
| Option 2 | 3      | 1      | 13    | 5      |
| Option 3 | 12     | 13     | 1      | 0      |

The following table and graph shows the average naturalness of the caption as rated by each respondant. This section gives the image and the corresponding caption and asks the respondant to rate how natural the sentence is on a scale from one to five.

![Mean Naturalness Score by Question](https://user-images.githubusercontent.com/35882267/80967263-ac856900-8ddb-11ea-9ccd-c29bbecfba97.png)

| C1     | C2     | C3     | C4     | Total  |
|:------:|:------:|:------:|:------:|:------:|
| 1.565  | 2.217  | 2.696	 | 3.609  | 2.522  |

The following table and graph shows the average helpfulness of the caption as rated by each respondant. This section gives the image and the corresponding caption and asks the respondant to rate how helpful the sentence is on a scale from one to five.

![Mean Helpfulness Score by Question](https://user-images.githubusercontent.com/35882267/80967262-abecd280-8ddb-11ea-8053-da858f83782b.png)

| C1     | C2     | C3     | C4     | Total  |
|:------:|:------:|:------:|:------:|:------:|
| 2.000  | 2.522  | 2.913	 | 3.522  | 2.739  |

Our survey is linked [here](https://docs.google.com/forms/d/1AErXKhsPgB2svVDI0yZcdJ-jh3XHiOOCuka5DMcZZtM/edit?ts=5eaef2a2). </br>

### Key Takeaways


### References
1. Falomir, Zoe, et al. "Describing images using qualitative models and description logics." Spatial Cognition & Computation 11.1 (2011): 45-74.
2. Elliott, Desmond, and Arjen de Vries. "Describing images using inferred visual dependency representations." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2015.
3. Socher, Richard, et al. "Grounded compositional semantics for finding and describing images with sentences." Transactions of the Association for Computational Linguistics 2 (2014): 207-218.
4. Ordonez, Vicente, Girish Kulkarni, and Tamara L. Berg. "Im2text: Describing images using 1 million captioned photographs." Advances in neural information processing systems. 2011.

### Files:
**Code:**
* grab_model.ipynb - An old notebook for testing the COCO dataset.
* maskrcnn.ipynb - Main Computer Vision code notebook. This notebook gets the data that is to be used by the nlpstuff notebook to generate sentences.
* helper_fcns.py - Computer Vision helper functions for maskrcnn notebook.
* nlpstuff.ipynb - A notebook containing only NLP functions. This notebook is what eventually generates the sentences.

**JSON:**
* results.json, failed.json - Old json files used to transfer data about one image to NLP notebook
* everything.json - Current json file that is used to transfer data about (currently) 200 images to NLP notebook.
* final_short_results.json - Final results from NLP notebook, shortened version (as some descriptions can be very long).
* final_long_results.json - Same as the previous file, just longer.
