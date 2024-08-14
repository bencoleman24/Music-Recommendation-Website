# Music Recommendation Website

This project is a web-based application that serves personalized song recommendations to users. Users can select the recommendation model type (e.g., popularity-based, cosine similarity, k-means, neural networks) and input their genre preferences or a song that they want a recommendations based off of. The system then generates tailored suggestions with corresponding song links, providing an smooth and interactive experience. 


 
<br />



__Features__:

* Model Selection: Choose from 4 recommendation models to tailor the experience.
* Feedback Mechanism: After viewing recommendations, users can provide feedback on the quality of recommendations, helping improve the system further.
* Spotify Song Links: Each song reccomendation comes with a spotify link taking the user right to the song

  
__Technical Stack__:

* Frontend: HTML, CSS, and JavaScript (I'm not a frontend developer, got some help here)
* Backend: Python (Flask, Pandas, Scikit-Learn, Tensorflow, Spotipy)

  
__Recommendation Limitations__:

The training data for the reccomendation models is relativley small (~350k rows) compared to what an extensive spotify dataset with every song would look like. This limits the potential of the models because it makes it difficult to add quality assurance filters like a minimum view count. Occasionally songs that are irrelevant or of low quality will be reccomended. Typically, music recommendation models rely on clustering algorithms that group listeners based on their preferences and suggest songs that similar listeners enjoy. However, I chose a different approach by using audio feature data to recommend songs. While this method might not be the most conventional or ideal, the results are still intriguing and entertaining. :)



  #### __Website Link__:  [mlmusicrec.com](http://www.mlmusicrec.com)
##### ^^^ Try the website yourself and get a recommendation ^^^ 
