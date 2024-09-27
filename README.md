 <h1>BERT Sentiment Analysis</h1>
 
  <p>
Fine-tuning and implementing the BERT model to classify sentiments in Google Play reviews.
  </p>

<br />

# Table of Contents

- [About the Project](#about-the-project)

  * [Tech Stack](#tech-stack)
  * [Features](#features)

- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)

- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  
## About the Project
This project leverages natural language processing (NLP) techniques to perform sentiment analysis on user reviews from the Google Play Store. By scraping data directly from the platform, we've compiled a comprehensive dataset of app reviews, which serves as the foundation for our analysis.

### Tech Stack

<details>
 
  <ul>
    <li><a href="https://pytorch.org">PyTorch</a></li>
    <li><a href="https://huggingface.co/docs/transformers/en/index">HuggingFace Transformers</a></li>
    <li><a href="https://seaborn.pydata.org">Seaborn</a></li>

  </ul>
</details>

### Features

- **Data Scraping**: Efficiently gathered thousands of user reviews using web scraping techniques, focusing on key metrics such as rating, review text, and timestamp.
- **Data Preprocessing**: Cleaned and prepared the dataset for analysis, including text normalization, tokenization, and removal of irrelevant information.
- **BERT Model Implementation**: Utilized the BERT classifier from the Hugging Face Transformers library to build a robust deep learning model capable of understanding context and nuances in language.

### Prerequisites

This project uses pip as package manager

```python
 pip install transformers
 pip install tensorflow[and-cuda]
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## Usage

-   **App Developers**: Gain insights into user feedback, identify common pain points, and enhance app features based on user sentiment.
    
-   **Market Researchers**: Analyze trends in user reviews across different apps, helping to understand market needs and preferences.
    
-   **Product Managers**: Make informed decisions on marketing strategies and product improvements based on real user opinions.
    
-   **Academic Research**: Serve as a resource for researchers studying sentiment analysis, NLP, and consumer behavior in the tech industry.

### Future Work

-   **Real-Time Analysis**: Integrate the model into a web application for real-time sentiment analysis of new reviews.
-   **Broader Data Sources**: Expand the scraping to include reviews from other platforms like the Apple App Store or social media.
-   **Enhanced Features**: Add functionality to analyze sentiment trends over time or correlate sentiments with app updates.


## License

Distributed under the Apache License. See LICENSE.txt for more information.

## Contact

Your Name - [LinkedIn](www.linkedin.com/in/prateekmp) - prateekmsoa@gmail.com

Project Link: [BERT-Sentiment-Analysis](https://github.com/ezahpizza/BERT-Sentiment-Analysis/tree/main)

## Acknowledgements

 - [Getting things done with PyTorch](https://leanpub.com/getting-things-done-with-pytorch)
 - [HuggingFace](https://huggingface.co/docs/transformers/index)

This project aims to provide insights into user sentiment towards mobile applications, aiding developers and marketers in understanding user feedback better. Happy coding!
