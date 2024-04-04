# Music Recommendation Website

This project is a web-based application that serves personalized song recommendations to users. Users can select the recommendation model type (e.g., popularity-based, cosine similarity, k-means, neural networks) and input their genre preferences or a song that they want a recommendations based off of. The system then generates tailored suggestions with corresponding song links, providing an smooth and interactive experience. 

## __Website Link__:  [mlmusicrec.com](http://www.mlmusicrec.com)
#### ^^^ Try the website yourself and get a recommendation ^^^ 

<br />
 
<br />



__Features__:

* Dynamic Model Selection: Choose from various recommendation algorithms to tailor the experience.
* Feedback Mechanism: After viewing recommendations, users can provide feedback on the quality of suggestions, helping improve the system further.
* User-friendly Interface: Sleek design with intuitive controls and clear feedback mechanisms.

  
__Technical Stack__:

* Frontend: HTML, CSS, and JavaScript
* Backend: Python (Flask, Pandas, Scikit-Learn, Tensorflow, Spotipy)

  
__Recommendation Limitations__:

The training data for the reccomendation models is relativley small (~350k rows) compared to what an extensive spotify dataset with every song would look like. This limits the potential of the models because it makes it difficult to add quality assurance filters like a minimum view count. Occasionally songs that are irrelevant or of low quality will be reccomended. 

__Future Enhancements__:

* External hosting for 24/7 accessibility.
* ~~Integrate backend of user feedback~~, add detailed analytics with feedback data.
*~~Replace the training dataset with one that is larger and more feature dense.~~
* Integrate more advanced recommendation algorithms and add improvements to existing models.
* Further enhance design/aesthetics of website.

  
