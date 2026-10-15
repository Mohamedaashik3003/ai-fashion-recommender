\# AI Fashion Studio



An AI-powered fashion recommendation project that helps users discover clothing combinations and explore virtual try-on. The application uses the IndoFashion dataset, which includes traditional Indian clothing for both men and women.



\## Features



\- \*\*Outfit Recommendations:\*\* Suggests clothing combinations using predefined category pairs.

\- \*\*Men's and Women's Fashion:\*\* Supports clothing categories for both men and women available in the dataset.

\- \*\*Multiple Outfit Styles:\*\* Explore different traditional Indian clothing combinations.

\- \*\*Recommendation Scores:\*\* Ranks sampled combinations using shared words in product titles.

\- \*\*CSV Export:\*\* Download generated outfit recommendations.

\- \*\*Virtual Try-On:\*\* Experimental interface for AI-based clothing try-on.

\- \*\*Garment Classification:\*\* Experimental garment-type detection.



\## Dataset



The project uses the IndoFashion dataset.



\- Approximately 106,166 metadata records

\- 15 traditional Indian clothing categories

\- Clothing for men and women

\- Metadata file: `data/indofashion\_metadata.csv`



The dataset images and metadata must be available locally for recommendations to work.



\## Technologies Used



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- scikit-learn

\- PyTorch

\- Transformers

\- IndoFashion dataset



\## Project Structure



```text

ai-fashion-recommender/

├── app/

│   └── app.py

├── data/

│   └── indofashion\_metadata.csv

├── models/

├── notebooks/

├── results/

├── src/

│   ├── real\_outfit\_recommender.py

│   ├── garment\_detector.py

│   └── prepare\_indofashion.py

└── tests/

```



\## Installation



Activate the project environment:



```bash

conda activate fashion-ai

```



Install the required packages if they are not already installed:



```bash

pip install streamlit pandas numpy scikit-learn matplotlib seaborn opencv-python pillow transformers

```



Install a compatible PyTorch build separately if needed.



\## Run the Application



From the project root, execute:



```bash

streamlit run app/app.py

```



Open the local URL displayed in the terminal.



\## Current Limitations



\- Outfit compatibility scores are experimental and based on product-title similarity.

\- Garment classification is experimental.

\- Full-outfit virtual try-on, including simultaneous upper- and lower-garment replacement, is under development.

\- Results depend on the capabilities and availability of the selected AI model.



\## Future Improvements



\- Improve garment detection and visual compatibility.

\- Support full-outfit virtual try-on for men and women.

\- Improve clothing-region preservation.

\- Enhance the interface and recommendation accuracy.



\## Project Status



\*\*In Development\*\*



An AI/ML project built with Python, Streamlit, and the IndoFashion dataset.



