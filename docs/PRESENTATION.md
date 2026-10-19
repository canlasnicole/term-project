# 10-minute presentation script

Speak slowly. Each part has a target time. Show the live site while you talk.
Key terms have the Korean word in brackets so you can say both if needed.

---

## Part 1. Introduction (0:00 - 1:00)

"Hello, my name is Nicole. My term project is an **AI and data-analysis integrated web service** [AI·데이터 분석 통합 웹 서비스].

The goal was not to practice each AI technique separately, but to put five AI functions into **one web service** and **deploy** [배포] it, so anyone can open it with a public URL.

The five modules are: a Gemini auto-reply system, data analysis and machine learning, CNN image classification, YOLO object detection, and RAG document question answering."

*(Show the Streamlit home page with the five tabs.)*

## Part 2. Architecture and MVC (1:00 - 2:30)

"The service has two parts.

The **frontend** [프론트엔드] is built with **Streamlit**. It is the screen the user sees.
The **backend** [백엔드] is built with **FastAPI**. It runs all the AI models and the data analysis.

They talk only through **HTTP requests** [HTTP 요청] with JSON data. Because of this, they can run on two separate servers, like in my **Docker** [도커] setup, or on one server, like my free cloud deployment. The code is the same; only one setting, `BACKEND_URL`, changes.

I followed the **MVC pattern** [MVC 패턴]:
- **Model** [모델]: `schemas.py` defines the shape of the data with Pydantic, and `storage.py` saves comments in SQLite.
- **View** [뷰]: `frontend/app.py`, the Streamlit screen. It never runs a model by itself.
- **Controller** [컨트롤러]: `main.py`, the FastAPI routes. It receives a request, checks it, calls the right service file, and returns JSON.

The AI logic is in separate **service files** [서비스 파일], one per module, so each part is easy to read and test."

*(Show the architecture diagram in README, then the `/docs` page.)*

## Part 3. Module 1: Gemini auto-reply (2:30 - 3:45)

"The first module simulates comments from social media.
A **webhook** [웹훅] is a URL that another platform calls automatically when something happens, for example a new YouTube comment.

My endpoint `POST /webhook/comment` receives the comment ID, platform, post ID, author and content. The backend saves it with status *pending*, sends it to the **Gemini API**, saves the AI reply, and changes the status to *replied*.

Let me send one with the webhook simulator."

*(Send a comment. Then send a Korean comment.)*

"Gemini answers in the same language as the comment. In the dashboard we see all comments, their status, and the detail JSON. The API key is stored only on the server, never in the browser."

## Part 4. Module 2: Data analysis and ML (3:45 - 6:00)

"For data analysis I used two datasets from *Python for Data Analysis*, Chapter 13: **Titanic** and **MovieLens 1M**.

First, **EDA** [탐색적 데이터 분석]. In Titanic, Age, Cabin and Embarked have **missing values** [결측치]. Women survived 74% versus men 19%. First class survived 63% versus third class 24%. Children also survived more.

In MovieLens, with one million ratings, I compared ratings by gender, age group and genre. For example, older users give higher ratings (3.51 for ages 18-24, 3.77 for 56+), and some movies split by gender: women rated *Dirty Dancing* 3.79 but men only 2.96.

Next, **model comparison** [모델 비교]. I compared six models: kNN, logistic regression, naive Bayes, decision tree, random forest and XGBoost. All use the same **preprocessing pipeline** [전처리 파이프라인]: fill missing values, scale numbers, and one-hot encode categories.

I used four **metrics** [평가 지표]:
- **Recall** [재현율]: of all real survivors, how many did we find?
- **F1-score**: balance of precision and recall.
- **AUROC**: how well the model ranks survivors above non-survivors, at all thresholds.
- **PR-AUC**: like AUROC, but focused on the positive class; good when classes are not balanced.

I used **5-fold cross-validation** [5겹 교차 검증] on 80% of train.csv."

*(Show the comparison table.)*

| Model | Recall | F1 | AUROC | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.718 | 0.756 | 0.887 | 0.863 |
| Random Forest | 0.674 | 0.732 | 0.872 | 0.853 |
| Logistic Regression | 0.696 | 0.724 | 0.856 | 0.835 |
| Naive Bayes | 0.755 | 0.729 | 0.840 | 0.775 |
| Decision Tree | 0.652 | 0.723 | 0.848 | 0.786 |
| kNN | 0.667 | 0.707 | 0.840 | 0.773 |

"**XGBoost** has the best F1, AUROC and PR-AUC, and the best average of the four metrics, so I selected it as the **final model** [최종 모델]. Naive Bayes has the highest recall, but its PR-AUC is much lower, so it makes more false alarms.

One important point: **`test.csv` has no `Survived` column**. It is the Kaggle competition file, so we cannot calculate any score on it. That is why I kept 20% of train.csv as a **hold-out set** [홀드아웃 검증 세트] that the model never saw. On it, XGBoost got about recall 0.65, F1 0.71, AUROC 0.82, PR-AUC 0.79, and accuracy 79%. Then I retrained on all of train.csv and made **predictions** [예측] for the 418 passengers in test.csv. You can download them as a CSV."

*(Predict one passenger with the form.)*

## Part 5. Module 3: CNN (6:00 - 7:00)

"The third module uses **ResNet-18**, a **pretrained** [사전 학습된] CNN trained on ImageNet with 1,000 classes. I did not train it; I only do **inference** [추론].

The image is resized to 224 by 224 and normalized with the ImageNet mean and standard deviation. I use `model.eval()` and `torch.no_grad()`, then **softmax** [소프트맥스] turns the outputs into probabilities."

*(Classify a dog image.)*

"To decide if it is a dog or a cat, I check the **class index** [클래스 번호]: 151 to 268 are dogs, 281 to 285 are cats. This is better than keyword matching, because the word 'cat' also matches 'catamaran', which is a boat."

## Part 6. Module 4: YOLO (7:00 - 7:45)

"The fourth module is **object detection** [객체 탐지] with **YOLOv8 Nano**, pretrained on COCO with 80 classes. Classification gives one label for the whole image; detection finds **many objects and their locations** with **bounding boxes** [바운딩 박스]."

*(Upload the dog-and-cat photo.)*

"Here it found a dog, a cat and a chair, each with a confidence score. The slider changes the minimum confidence."

## Part 7. Module 5: RAG (7:45 - 9:00)

"The last module is **RAG**, retrieval-augmented generation [검색 증강 생성]. I re-implemented an n8n RAG workflow with **LangChain**. Each n8n node became one step in code:
the document is loaded, split into **chunks** [청크], and each chunk becomes an **embedding** [임베딩], a list of numbers that represents meaning. I store them in **FAISS**, a **vector database** [벡터 데이터베이스].

For embeddings I use the **Gemini embedding model**, so no extra model has to fit in my free server's memory.

When a user asks a question, the question also becomes an embedding, FAISS finds the top-k most similar chunks, and Gemini writes the answer **using only those chunks**."

*(Upload a PDF, ask a question, open the source chunks.)*

"The sources show where the answer came from, so the user can check it. If the answer is not in the document, it says so, which reduces **hallucination** [환각]."

## Part 8. Deployment and wrap-up (9:00 - 10:00)

"For **deployment**, I had a real problem to solve. Hugging Face now needs a paid plan for Docker apps, and Render's free plan has only 512 MB of memory, which is not enough for PyTorch and YOLO.

So I deployed everything on **Streamlit Community Cloud**, connected to GitHub, with the Gemini key stored as a **secret** [시크릿]. When the app starts, it runs the FastAPI server in a **background thread** [백그라운드 스레드], and the Streamlit screen still calls it through HTTP. To fit in the free memory, I used CPU-only PyTorch, **lazy loading** [지연 로딩] so each model loads only on its first request, Gemini embeddings instead of a local model, and a saved summary of the one-million-row MovieLens analysis.

For the Docker submission, the same code runs as two separate containers with docker compose.

In summary: five AI modules, one FastAPI backend, one Streamlit frontend, MVC structure, and a public URL that anyone can test. Thank you. I am happy to answer questions."

---

# 10 questions the professor may ask

**1. FastAPI and Streamlit run on the same cloud server. Are they really separate?**
Yes. They are two programs that talk only through HTTP. Streamlit sends requests to `http://127.0.0.1:8000`, the same way it would call a remote server. With Docker they run as two containers, and only the `BACKEND_URL` setting changes, not the code. I used one server because free Docker hosting with enough memory is no longer available.

**2. Where is MVC in your code?**
Model = `schemas.py` and `storage.py`. View = `frontend/app.py`. Controller = `main.py`. The service files hold the AI logic that the controller calls.

**3. test.csv has no labels. How do you report test performance?**
I cannot score test.csv. I report performance on a 20% hold-out split of train.csv that was never used for training or model selection. test.csv is used only for predictions, like in the Kaggle competition.

**4. Why did you choose XGBoost?**
It had the highest average of the four metrics in 5-fold cross-validation, and the best F1, AUROC and PR-AUC. Naive Bayes had higher recall but much lower PR-AUC, so it predicts "survived" too often.

**5. What is the difference between AUROC and PR-AUC?**
AUROC measures ranking quality using true-positive rate and false-positive rate. PR-AUC uses precision and recall, so it focuses on the positive class. When positives are fewer (here 38% survived), PR-AUC shows problems that AUROC can hide.

**6. How do you prevent data leakage?**
All preprocessing (filling missing values, scaling, encoding) is inside a scikit-learn Pipeline, so it is fitted only on the training folds in each cross-validation step. The hold-out set is split before any model selection.

**7. Why use class indices for dog/cat instead of keywords?**
Keywords give false matches, for example "cat" in "catamaran". ImageNet classes 151-268 are dogs and 281-285 are domestic cats, so the index check is exact.

**8. What is the difference between ResNet-18 and YOLO?**
ResNet-18 is a classifier: one label for the whole image. YOLO is a detector: many objects, each with a class, confidence and box position, in one pass.

**9. How does RAG reduce hallucination, and why FAISS?**
Gemini is told to answer only from the retrieved chunks and to say when the answer is missing; the sources are shown so users can check. FAISS is a fast, free vector search library that runs in memory, which is enough for one document.

**10. What would you improve with more time?**
A persistent database (e.g. PostgreSQL) so comments survive restarts, webhook signature checking for security, hyperparameter tuning (GridSearchCV) for XGBoost, saving multiple documents for RAG, and a real platform connection such as the YouTube Data API.
