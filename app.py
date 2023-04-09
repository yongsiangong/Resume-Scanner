import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import openai

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words += ['data', 'experience', 'business', 'team', 'work', 'skill', 'solution', 'system', 'project',
                       'requirement', 'management', 'support', 'tool', 'technology', 'engineering', 'knowledge', 'design',
                       'science', 'development', 'working', 'strong', 'technical', 'product', 'process', 'year', 'using',
                       'ability', 'platform', 'service', 'good', 'develop', 'build', 'etc', 'job', 'role', 'new',
                       'stakeholder', 'including', 'responsibility', 'client', 'application', 'problem', 'related',
                       'understanding', 'user', 'opportunity', 'environment', 'big', 'provide', 'ensure', 'quality',
                       'architecture', 'across', 'end', 'communication', 'key', 'building', 'need', 'source', 'performance',
                       'use', 'candidate', 'customer', 'required', 'information', 'company', 'able', 'understand',
                       'processing', 'infrastructure', 'like', 'relevant', 'industry', 'high', 'least', 'singapore',
                       'operation', 'aws', 'report', 'research', 'based', 'identify', 'manage', 'well', 'reporting', 'hand',
                       'apply', 'within', 'various', 'improve', 'maintain', 'language', 'implement', 'practice',
                       'implementation', 'large', 'testing', 'drive', 'security', 'issue', 'create', 'plus', 'production',
                       'global', 'delivery', 'time', 'field', 'driven', 'developing', 'strategy', 'deep', 'one', 'best',
                       'internal', 'com', 'lead', 'advanced', 'integration', 'ea', 'part', 'set', 'responsible',
                       'functional', 'excellent', 'looking', 'preferred', 'technique', 'perform', 'scale', 'complex',
                       'deliver', 'code', 'value', 'level', 'multiple', 'qualification', 'solving', 'decision', 'partner',
                       'standard', 'test', 'market', 'area', 'people', 'help', 'make', 'improvement', 'leading', 'world',
                       'real', 'highly', 'join', 'designing', 'impact', 'collaborate', 'expertise', 'enterprise',
                       'function', 'dashboard', 'please', 'group', 'closely']
stop_words += ['digital', 'ha', 'also', 'review', 'minimum', 'existing', 'description', 'policy', 'training',
                       'control', 'learn', 'must', 'critical', 'career', 'deployment', 'tiktok', 'proficient', 'senior',
                       'effectively', 'concept', 'result', 'activity', 'personal', 'office', 'day', 'self', 'advantage',
                       'position', 'communicate', 'continuous', 'managing', 'task', 'expert', 'member', 'change',
                       'equivalent', 'external', 'familiar', 'initiative', 'solve', 'effective', 'open', 'domain', 'meet',
                       'transformation', 'plan', 'following', 'different', 'experienced', 'operational', 'professional',
                       'would', 'innovative', 'methodology', 'trend', 'monitoring', 'implementing', 'storage', 'capability',
                       'manager', 'asset', 'case', 'challenge', 'scripting', 'creating', 'document', 'purpose', 'benefit',
                       'organization', 'growth', 'detail', 'written', 'core', 'structured', 'mining', 'maintenance',
                       'track', 'cross', 'applicant', 'exposure', 'investment', 'diverse', 'bachelorâ', 'structure', 'sg',
                       'culture', 'assist', 'strategic', 'firm', 'community', 'participate', 'employee', 'feature',
                       'similar', 'discipline', 'health', 'life', 'interested', 'approach', 'shortlisted', 'sale',
                       'ingestion', 'planning', 'variety', 'efficiency', 'conduct', 'monitor', 'proven', 'pte', 'making',
                       'define', 'architect', 'leadership', 'privacy', 'vendor', 'take', 'server', 'healthcare',
                       'familiarity', 'individual', 'batch', 'master', 'include', 'number', 'supporting', 'range', 'ltd',
                       'procedure', 'successful', 'deploy', 'multi', 'record', 'committed']
stop_words += ['goal', 'idea', 'full', 'bring', 'program', 'resume', 'metric', 'fast', 'offer', 'independently',
                       'distributed', 'http', 'non', 'dynamic', 'may', 'way', 'pattern', 'background', 'efficient',
                       'modern', 'wide', 'computing', 'image', 'salary', 'depth', 'objective', 'enable', 'preferably',
                       'method', 'player', 'centre', 'devops', 'passionate', 'actionable', 'future', 'providing',
                       'contribute', 'access', 'focus', 'proficiency', 'relationship', 'right', 'maintaining', 'cd',
                       'marketing', 'solid', 'license', 'outcome', 'extract', 'predictive', 'registration', 'ci',
                       'motivated', 'location', 'achieve', 'mission', 'flow', 'analyse', 'collection', 'public', 'tech',
                       'success', 'limited', 'interpersonal', 'transform', 'notified', 'evaluate', 'leader', 'added', 'www',
                       'employment', 'insurance', 'better', 'unit', 'robust', 'script', 'prior', 'inclusive', 'resource',
                       'cv', 'center', 'used', 'demonstrated', 'format', 'basic', 'handle', 'leverage', 'video',
                       'translate', 'inspire', 'passion', 'specification', 'stream', 'collaboration', 'troubleshooting',
                       'gcp', 'current', 'innovation', 'present', 'includes', 'around', 'validation', 'appropriate',
                       'write', 'enhance', 'email', 'ad', 'department', 'contract', 'writing', 'ecosystem', 'tuning',
                       'effort', 'integrity', 'posse', 'matter', 'ensuring', 'edge']
stop_words += ['assessment', 'space', 'necessary', 'thinking', 'deploying', 'certification', 'potential', 'line',
                       'finding', 'google', 'event', 'available', 'expected', 'comfortable', 'collect', 'store', 'request',
                       'regret', 'subject', 'order', 'growing', 'party', 'excellence', 'point', 'fraud', 'establish',
                       'consumer', 'execution', 'grow', 'timely', 'portfolio', 'come', 'commercial', 'country', 'principle',
                       'needed', '10', 'optimize', 'reg', 'overall', 'scalability', 'applied', 'classification',
                       'deliverable', 'generate', 'send', 'object', 'optimizing', 'assigned', 'mapping', 'warehousing',
                       'asia', 'beyond', 'comprehensive', 'coordinate', 'credit', 'employer', 'reference', 'licence',
                       'class', 'shell', 'delivering', 'great', 'term', 'youâ', 'get', 'scope', 'operating', 'together',
                       'act', 'commerce', 'verbal', 'guidance', 'action', 'execute', 'accuracy', 'diploma', 'web',
                       'creative', 'challenging', 'collaborative', 'latest', 'explore', 'facilitate', 'unique',
                       'resolution', 'consulting', 'enhancement', 'address', 'aim', 'analytic', 'involved', 'state',
                       'prepare', 'online', 'intelligent', 'performing', 'duty', 'glue', 'sector', 'personnel', 'weâ',
                       'form', 'globally', 'creativity', 'provider', 'international', 'factory', 'hoc', 'component',
                       'workplace', 'base']

da_score_dict = {'analysis': 0.06451612903225806, 'analytics': 0.06236559139784946, 'analyst': 0.060215053763440864, 'sql': 0.05806451612903226, 'insight': 0.05591397849462366, 'degree': 0.053763440860215055, 'tableau': 0.05161290322580645, 'analytical': 0.04946236559139785, 'python': 0.047311827956989246, 'risk': 0.04516129032258064, 'model': 0.043010752688172046, 'bi': 0.04086021505376344, 'visualization': 0.03870967741935484, 'financial': 0.03655913978494624, 'computer': 0.034408602150537634, 'database': 0.03225806451612903, 'power': 0.030107526881720432, 'learning': 0.02795698924731183, 'intelligence': 0.025806451612903226, 'excel': 0.023655913978494623, 'statistic': 0.021505376344086023, 'statistical': 0.01935483870967742, 'finance': 0.017204301075268817, 'bank': 0.015053763440860216, 'query': 0.012903225806451613, 'power bi': 0.010752688172043012, 'presentation': 0.008602150537634409, 'analyze': 0.0064516129032258064, 'programming': 0.004301075268817204, 'documentation': 0.002150537634408602}
ds_score_dict = {'learning': 0.06451612903225806, 'model': 0.06236559139784946, 'machine': 0.060215053763440864, 'machine learning': 0.05806451612903226, 'risk': 0.05591397849462366, 'ai': 0.053763440860215055, 'analytics': 0.05161290322580645, 'ml': 0.04946236559139785, 'computer': 0.047311827956989246, 'algorithm': 0.04516129032258064, 'python': 0.043010752688172046, 'analysis': 0.04086021505376344, 'software': 0.03870967741935484, 'degree': 0.03655913978494624, 'scientist': 0.034408602150537634, 'insight': 0.03225806451612903, 'statistical': 0.030107526881720432, 'engineer': 0.02795698924731183, 'programming': 0.025806451612903226, 'framework': 0.023655913978494623, 'statistic': 0.021505376344086023, 'sql': 0.01935483870967742, 'analytical': 0.017204301075268817, 'cloud': 0.015053763440860216, 'vision': 0.012903225806451613, 'pipeline': 0.010752688172043012, 'learning model': 0.008602150537634409, 'quantitative': 0.0064516129032258064, 'intelligence': 0.004301075268817204, 'mathematics': 0.002150537634408602}
de_score_dict = {'pipeline': 0.06451612903225806, 'analytics': 0.06236559139784946, 'cloud': 0.060215053763440864, 'sql': 0.05806451612903226, 'engineer': 0.05591397849462366, 'database': 0.053763440860215055, 'python': 0.05161290322580645, 'azure': 0.04946236559139785, 'etl': 0.047311827956989246, 'spark': 0.04516129032258064, 'computer': 0.043010752688172046, 'model': 0.04086021505376344, 'software': 0.03870967741935484, 'degree': 0.03655913978494624, 'analysis': 0.034408602150537634, 'learning': 0.03225806451612903, 'hadoop': 0.030107526881720432, 'warehouse': 0.02795698924731183, 'degree computer': 0.025806451612903226, 'lake': 0.023655913978494623, 'programming': 0.021505376344086023, 'machine': 0.01935483870967742, 'machine learning': 0.017204301075268817, 'java': 0.015053763440860216, 'framework': 0.012903225806451613, 'analytical': 0.010752688172043012, 'network': 0.008602150537634409, 'modelling': 0.0064516129032258064, 'documentation': 0.004301075268817204, 'oracle': 0.002150537634408602}


def recommender(resume_pdf, pred_model):
    reader = PdfReader(resume_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text = text.strip()
    new_text = ""
    for char in text:
        if char.isalpha() or char.isspace():
            new_text += char.lower()
    resume_text = " ".join(new_text.split())

    prediction = pd.DataFrame({'resume': resume_text}, index=[0])

    cleaning_regex = '[^\w\s]|\n|\n\n'
    prediction['resume'] = prediction['resume'].str.lower().replace(cleaning_regex, " ", regex=True).str.strip()
    prediction['resume_cleaned'] = prediction['resume'].apply(lambda x: x.split())
    prediction['resume_cleaned'] = prediction['resume_cleaned'].apply(
        lambda x: [lemmatizer.lemmatize(item) for item in x])
    prediction['resume_cleaned'] = prediction['resume_cleaned'].apply(
        lambda x: [item for item in x if item not in stop_words])

    def scoring_mod(row, score_dict):
        score = 0
        missing_skill = []
        for k, v in score_dict.items():
            if k in row:
                score += v
            else:
                missing_skill.append(k)
        return score, missing_skill

    resume_text = prediction['resume'][0]

    prediction['da_score'], prediction['da_missing_skills'] = zip(*prediction['resume_cleaned'].apply(scoring_mod, score_dict=da_score_dict))
    prediction['ds_score'], prediction['ds_missing_skills'] = zip(*prediction['resume_cleaned'].apply(scoring_mod, score_dict=ds_score_dict))
    prediction['de_score'], prediction['de_missing_skills'] = zip(*prediction['resume_cleaned'].apply(scoring_mod, score_dict=de_score_dict))

    scores = prediction[['da_score', 'de_score', 'ds_score']]
    prediction['rec'] = pred_model.predict(scores)[0]

    return prediction

model_training_set = pd.read_csv('streamlit.csv')

seed = 42
new_X = model_training_set.drop('field', axis=1)
new_y = model_training_set['field']

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.3, random_state=seed)

svc = SVC(random_state=42)
svc.fit(X_train, y_train)


tab1, tab2 = st.tabs(['Resume Scanner', 'Resume Filter'])

with tab1:
    st.header("Resume Scanner")
    st.subheader("1. Instructions:")
    st.write("To determine your compatibility for the roles of data analyst, data engineer, and data scientist, upload your resume to receive a compatibility score, recommended job category and missing keywords/skills in yor resume. The higher your score, the more suitable you are for the corresponding job category.")
    st.subheader("2. Upload Resume:")
    uploaded_file = st.file_uploader("Only pdf allowed:")
    if uploaded_file is not None:

        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        text = text.strip()
        new_text = ""
        for char in text:
            if char.isalpha() or char.isspace():
                new_text += char.lower()
        resume_text = " ".join(new_text.split())

        prediction = pd.DataFrame({'resume': resume_text}, index=[0])

        cleaning_regex = '[^\w\s]|\n|\n\n'
        prediction['resume'] = prediction['resume'].str.lower().replace(cleaning_regex, " ", regex=True).str.strip()
        prediction['resume_cleaned'] = prediction['resume'].apply(lambda x: x.split())
        prediction['resume_cleaned'] = prediction['resume_cleaned'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
        prediction['resume_cleaned'] = prediction['resume_cleaned'].apply(lambda x: [item for item in x if item not in stop_words])


        prediction = recommender(uploaded_file, svc)
        st.subheader('3. Compatibility Scores and Prediction')
        # st.write(prediction[['da_score','ds_score','de_score']])
        st.write(f"The most compatible job based on your resume is {prediction['rec'][0].title()}")
        st.bar_chart(prediction[['da_score','ds_score','de_score']].T.rename(columns={0: 'Scores'}))

        if prediction['rec'][0] == 'data analyst':
            st.subheader(':blue[Data Analyst]')
            st.image("da_b.png")
            st.write(f":blue[Prediction score: {round(prediction['da_score'][0], 2)}]")
            st.write(f":blue[Missing keywords/skills: {', '.join(prediction['da_missing_skills'][0])}]")

            st.subheader('Data Scientist')
            st.image("ds_w.png")
            st.write(f"Prediction score: {round(prediction['ds_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {', '.join(prediction['ds_missing_skills'][0])}")

            st.subheader('Data Engineer')
            st.image("de_w.png")
            st.write(f"Prediction score: {round(prediction['de_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {', '.join(prediction['de_missing_skills'][0])}")

        elif prediction['rec'][0] == 'data scientist':
            st.subheader(':blue[Data Scientist]')
            st.image("ds_b.png")
            st.write(f":blue[Prediction score: {round(prediction['ds_score'][0], 2)}]")
            st.write(f":blue[Missing keywords/skills: {', '.join(prediction['ds_missing_skills'][0])}]")

            st.subheader('Data Analyst')
            st.image("da_w.png")
            st.write(f"Prediction score: {round(prediction['da_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {', '.join(prediction['da_missing_skills'][0])}")

            st.subheader('Data Engineer')
            st.image("de_w.png")
            st.write(f"Prediction score: {round(prediction['de_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {', '.join(prediction['de_missing_skills'][0])}")

        elif prediction['rec'][0] == 'data engineer':
            st.subheader(':blue[Data Engineer]')
            st.image("de_b.png")
            st.write(f":blue[Prediction score: {round(prediction['de_score'][0], 2)}]")
            st.write(f":blue[Missing keywords/skills: {', '.join(prediction['de_missing_skills'][0])}]")

            st.subheader('Data Analyst')
            st.image("da_w.png")
            st.write(f"Prediction score: {round(prediction['da_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {', '.join(prediction['da_missing_skills'][0])}")

            st.subheader('Data Scientist')
            st.image("ds_w.png")
            st.write(f"Prediction score: {round(prediction['ds_score'][0], 2)}")
            st.write(f"Missing keywords/skills: {','.join(prediction['ds_missing_skills'][0])}")

        st.subheader("4. Upskilling")
        upskilling_container = st.container()
        with upskilling_container:
            upskilling_input = st.text_area("Type in the skill you would like to learn:")

            if len(upskilling_input) != 0:
                openai.api_key = st.secrets['api']
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{"role": "user", "content": f"Recommend 5 online courses to learn: {upskilling_input}" }]
                                                        )
                output_text = response['choices'][0]['message']['content']
                st.subheader("5. Recommended Courses")
                st.write(output_text)



with tab2:
    st.header("Resume Filter")
    st.subheader("1. Instructions:")
    st.write("You can upload multiple resumes to receive compatibility scores for the positions of data analyst, data engineer, and data scientist. In addition, you have the option to filter the resumes based on relevant keywords.")
    st.subheader("2. Upload Resumes:")
    uploaded_files = st.file_uploader("Only pdfs allowed", accept_multiple_files=True)
    if uploaded_files:
        df_list = []
        for uf in uploaded_files:
            df = recommender(uf, svc)
            df['file'] = uf.name
            df_list.append(df)
        all_df = pd.concat(df_list, axis=0)
        all_df_final = all_df[['file', 'rec', 'da_score', 'ds_score', 'de_score']]

        features = list(set(list(da_score_dict.keys()) + list(ds_score_dict.keys()) + list(de_score_dict.keys())))

        def filter_dataframe(df):
            st.subheader("3. Filtered Results")
            modify = st.checkbox("Check the box to enable search")

            if not modify:
                return df[['file', 'rec', 'da_score', 'ds_score', 'de_score']]

            else:
                df = df.copy()
                modification_container = st.container()
                with modification_container:
                    user_input = st.text_area("Enter required keywords/skills separated by comma:")
                    user_input_changed = [x.lower() for x in user_input.replace(" ","").replace("\n","").split(",")]

                    if len(user_input)!=0:
                        df['resume_cleaned_str'] = df['resume_cleaned'].apply(lambda x: " ".join(x))
                        filtered = df[df['resume_cleaned'].apply(lambda x: set(user_input_changed).issubset(x))][['file', 'rec', 'da_score', 'ds_score', 'de_score']]
                        del user_input
                        return filtered
                    else:
                        return df[['file', 'rec', 'da_score', 'ds_score', 'de_score']]


        st.write(filter_dataframe(all_df))



