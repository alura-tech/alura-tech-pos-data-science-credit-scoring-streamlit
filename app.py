import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from joblib import load
import streamlit as st


train_original = pd.read_csv('https://github.com/vqrca/teste/blob/main/train.csv?raw=true', sep=',')

test_original = pd.read_csv('https://github.com/vqrca/teste/blob/main/test.csv?raw=true', sep=',')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)

def data_split(df, test_size):
    SEED = 1561651
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_df, test_df = data_split(full_data, 0.2)

train_df_copy = train_df.copy()

def value_cnt_norm_cal(df,feature):
    '''
    Function to calculate the count of each value in a feature and normalize it
    '''
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['ID']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class OneHotWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,one_hot_enc_ft = ['Family_status', 'Housing_type', 'Income_type', 'Occupation_type']):                                      
                                                                               
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # function to one hot encode the features in one_hot_enc_ft
            def one_hot_enc(df,one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                # get the result of the one hot encoding columns names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                # change the array of the one hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),columns=feat_names_one_hot_enc,index=df.index)
                return df
            # function to concatenat the one hot encoded features with the rest of features that were not encoded
            def concat_with_rest(df,one_hot_enc_df,one_hot_enc_ft):
                # get the rest of the features
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]],axis=1)
                return df_concat
            # one hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df,self.one_hot_enc_ft)
            # returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(df,one_hot_enc_df,self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df

class OrdinalFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_enc_ft = ['Education_type']):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Education_type' in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Age', 'Total_income', 'Num_family', 'Years_employed']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'BAD' in df.columns:
            # smote function to oversample the minority class to fix the imbalance data
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'BAD'],df['BAD'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("BAD is not in the dataframe")
            return df

def full_pipeline(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotWithFeatNames()),
        ('ordinal_feature', OrdinalFeatNames()),
        ('min_max_scaler', MinMaxWithFeatNames()),
        ('oversample', Oversample())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline


############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #3e0f7d}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> ALURABANK - Formulário para Solicitação de Cartão de Crédito</h1>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')


st.text_input('Digite o seu nome completo')

# Age input slider
st.write("""
### Idade
""")
input_age = float(st.slider('Selecione a sua idade', value=42, min_value=18, max_value=70, step=1))

# Education level dropdown
st.write("""
### Nível de escolaridade
""")
edu_level_values = list(value_cnt_norm_cal(full_data,'Education_type').index)
edu_level_key = ['Secondary / secondary special','Higher education','Incomplete higher','Lower secondary','Academic degree']
edu_level_dict = dict(zip(edu_level_key,edu_level_values))
input_edu_level_key = st.selectbox('Selecione qual é o seu nível de formação', edu_level_key)
input_edu_level_val = edu_level_dict.get(input_edu_level_key)

# Marital status input dropdown
st.write("""
### Estado civil
""")
marital_status_values = list(value_cnt_norm_cal(full_data,'Family_status').index)
marital_status_key = ['Married', 'Single/not married', 'Civil marriage', 'Separated', 'Widow']
marital_status_dict = dict(zip(marital_status_key,marital_status_values))
input_marital_status_key = st.selectbox('Selecione o seu estado civil', marital_status_key)
input_marital_status_val = marital_status_dict.get(input_marital_status_key)

# Family member count
st.write("""
### Família
""")
fam_member_count = float(st.selectbox('Selecione quantos membros tem na sua família', [1,2,3,4,5,6,7,8,9,10,11,12]))

st.write("""
## Bens
""")

# Car ownship input
#st.write("""
## Car ownship
#""")
input_car_ownship = st.radio('Você possui um automóvel?',['Sim','Não'], index=0)
input_car_ownship_dict = {'Sim': 1, 'Não':0}
input_car_ownship_val = input_car_ownship_dict.get(input_car_ownship)

# Property ownship input
#st.write("""
## Property ownship
#""")
input_prop_ownship = st.radio('Você possui uma propriedade?',['Sim','Não'], index=0)
input_prop_ownship_dict = {'Sim': 1, 'Não':0}
input_prop_ownship_val = input_prop_ownship_dict.get(input_prop_ownship)

# Dwelling type dropdown
st.write("""
### Tipo de residência
""")
dwelling_type_values = list(value_cnt_norm_cal(full_data,'Housing_type').index)
dwelling_type_key = ['House / apartment', 'Live with parents', 'Municipal apartment ', 'Rented apartment', 'Office apartment', 'Co-op apartment']
dwelling_type_dict = dict(zip(dwelling_type_key,dwelling_type_values))
input_dwelling_type_key = st.selectbox('Selecione o seu tipo de residência', dwelling_type_key)
input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)

# Unemployment status dropdown
st.write("""
### Situação de desemprego
""")
input_unemployment_status = st.radio('Você está trabalhando no momento?',['Sim','Não'], index=0)
input_unemployment_status_dict = {'Sim': 1, 'Não':0}
input_unemployment_status_val = input_unemployment_status_dict.get(input_unemployment_status)

# Employment status dropdown
st.write("""
### Situação de emprego
""")
employment_status_values = list(value_cnt_norm_cal(full_data,'Income_type').index)
employment_status_key = ['Working','Commercial associate','Pensioner','State servant','Student']
employment_status_dict = dict(zip(employment_status_key,employment_status_values))
input_employment_status_key = st.selectbox('Selecione o seu status de trabalho atual', employment_status_key)
input_employment_status_val = employment_status_dict.get(input_employment_status_key)

# Position
st.write("""
### Ocupação
""")
job_status_values = list(value_cnt_norm_cal(full_data,'Occupation_type').index)
job_status_key = ['Other', 'Security staff', 'Sales staff', 'Accountants', 'Laborers',
       'Managers', 'Drivers', 'Core staff', 'High skill tech staff',
       'Cleaning staff', 'Private service staff', 'Cooking staff',
       'Low-skill Laborers', 'Medicine staff', 'Secretaries',
       'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff']
job_status_dict = dict(zip(job_status_key,job_status_values))
input_job_status_key = st.selectbox('Selecione qual é a sua ocupação atual', job_status_key)
input_job_status_val = job_status_dict.get(input_job_status_key)

# Employment length input slider
st.write("""
### Experiência
""")
input_employment_length = float(st.slider('Selecione o seu tempo de experiência em anos', value=6, min_value=0, max_value=30, step=1))

# Income
st.write("""
### Rendimentos
""")
input_income = float(st.text_input('Digite o seu rendimento anual (em reais) e pressione ENTER para confirmar',0))

# Work phone input
st.write("""
### Telefone corporativo
""")
input_work_phone = st.radio('Você tem um telefone corporativo?',['Sim','Não'], index=0)
work_phone_dict = {'Sim': 1, 'Não':0}
work_phone_val = work_phone_dict.get(input_work_phone)

# Phone input
st.write("""
### Telefone fixo
""")
input_phone = st.radio('Você tem um telefone fixo?',['Sim','Não'], index=0)
work_dict = {'Sim': 1, 'Não':0}
phone_val = work_dict.get(input_phone)

st.text_input('Digite um número de telefone-fixo ou celular')

# Email input
st.write("""
### Email
""")
input_email = st.radio('Você tem um email?',['Sim','Não'], index=0)
email_dict = {'Sim': 1, 'Não':0}
email_val = email_dict.get(input_email)

st.text_input('Digite o seu email')
 

# list of all the input variables
new_profile = [0, # ID
                    input_car_ownship_val, # car ownership
                    input_prop_ownship_val, # property ownership
                    work_phone_val, # Work phone
                    phone_val, # Phone
                    email_val,  # Email
                    input_unemployment_status_val, # unemployment status
                    fam_member_count,  # Family member count
                    input_income, # Income
                    input_age, # Age
                    input_employment_length, # Employment length
                    input_employment_status_val, # Employment status
                    input_edu_level_val, # Education level
                    input_marital_status_val, # Marital status
                    input_dwelling_type_val, # Dwelling type                                                  
                    input_job_status_val, # job title
                     0 # target set to 0 as a placeholder
                    ]

profile_to_predict_df = pd.DataFrame([new_profile],columns=train_df_copy.columns)

train_new_profile = pd.concat([train_df_copy,profile_to_predict_df],ignore_index=True)

train_new_profile = full_pipeline(train_new_profile)

profile_to_pred = train_new_profile.drop(['BAD'], axis=1)

# Predictions  
if st.button('Enviar'):
    model = joblib.load('objetos/randomforest.joblib')
    final_pred = model.predict(profile_to_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons()
    else:
        st.error('### Infelizmente, não podemos liberar crédito para você agora!')
 