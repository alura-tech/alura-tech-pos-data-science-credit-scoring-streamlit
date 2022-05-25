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


# Carregando os dados de treino e teste

train_original = pd.read_csv('https://raw.githubusercontent.com/alura-tech/alura-tech-pos-data-science-credit-scoring-streamlit/main/train.csv', sep=',')

test_original = pd.read_csv('https://raw.githubusercontent.com/alura-tech/alura-tech-pos-data-science-credit-scoring-streamlit/main/test.csv', sep=',')

# Concatenando os dados de treino e teste

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)

# Separando os dados em treino e teste

def data_split(df, test_size):
    SEED = 1561651
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_df, test_df = data_split(full_data, 0.2)

train_df_copy = train_df.copy()

# Função para calcular a contagem de cada valor e normalizá-los

def value_cnt_norm_cal(df,feature):
 
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Contagem', 'Frequência (%)']
    return ftr_value_cnt_concat

# Classes para pipeline

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['ID_Cliente']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class OneHotEncodingNames(BaseEstimator,TransformerMixin):
    def __init__(self,OneHotEncoding = ['Estado_civil', 'Moradia', 'Categoria_de_renda', 
                                        'Ocupacao']):                                      
                                                                               
        self.OneHotEncoding = OneHotEncoding

    def fit(self,df):
        return self

    def transform(self,df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            # função para one-hot-encoding das features
            def one_hot_enc(df,OneHotEncoding):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[OneHotEncoding])
                # obtendo o resultado dos nomes das colunas
                feature_names = one_hot_enc.get_feature_names(OneHotEncoding)
                # mudando o array do one hot encoding para um dataframe com os nomes das colunas
                df = pd.DataFrame(one_hot_enc.transform(df[self.OneHotEncoding]).toarray(),
                                  columns= feature_names,index=df.index)
                return df

            # função para concatenar as features com aquelas que não passaram pelo one-hot-encoding
            def concat_with_rest(df,one_hot_enc_df,OneHotEncoding):              
                # get the rest of the features
                outras_features = [feature for feature in df.columns if feature not in OneHotEncoding]
                # concaternar o restante das features com as features que passaram pelo one-hot-encoding
                df_concat = pd.concat([one_hot_enc_df, df[outras_features]],axis=1)
                return df_concat

            # one hot encoded dataframe
            df_OneHotEncoding = one_hot_enc(df,self.OneHotEncoding)

            # retorna o dataframe concatenado
            df_full = concat_with_rest(df, df_OneHotEncoding,self.OneHotEncoding)
            return df_full

class OrdinalFeature(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_feature = ['Grau_escolaridade']):
        self.ordinal_feature = ordinal_feature
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Grau_escolaridade' in df.columns:
            ordinal_encoder = OrdinalEncoder()
            df[self.ordinal_feature] = ordinal_encoder.fit_transform(df[self.ordinal_feature])
            return df
        else:
            print('Grau_escolaridade não está no DataFrame')
            return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Idade', 'Rendimento_anual', 'Tamanho_familia', 'Anos_empregado']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print('Uma ou mais features não estão no DataFrame')
            return df

class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Mau' in df.columns:
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Mau'],df['Mau'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print('Mau não está no DataFrame')
            return df

#Pipeline

def full_pipeline(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMaxWithFeatNames()),
        ('oversample', Oversample())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline


############################# Streamlit ############################

st.markdown('<style>div[role="listbox"] ul{background-color: #3e0f7d}; </style>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align: left; '> Formulário para Solicitação de Cartão de Crédito</h2>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

# Idade
st.write("""
### Idade
""")
input_idade = float(st.slider('Selecione a sua idade', value=42, min_value=18, max_value=70, step=1))

# Grau de escoloridade
st.write("""
### Nível de escolaridade
""")
grau_escolaridade_values = list(value_cnt_norm_cal(full_data,'Grau_escolaridade').index)
grau_escolaridade_key = ['Ensino fundamental','Ensino médio','Ensino superior incompleto','Ensino superior','Pós-graduação']
grau_escolaridade_dict = dict(zip(grau_escolaridade_key, grau_escolaridade_values))
input_grau_escolaridade_key = st.selectbox('Selecione qual é o seu nível de formação', grau_escolaridade_key)
input_grau_escolaridade_val = grau_escolaridade_dict.get(input_grau_escolaridade_key)

# Estado civil
st.write("""
### Estado civil
""")
estado_civil_values = list(value_cnt_norm_cal(full_data,'Estado_civil').index)
estado_civil_key = ['União-estável', 'Casado', 'Solteiro', 'Divorciado', 'Viúvo']
estado_civil_dict = dict(zip(estado_civil_key, estado_civil_values))
input_estado_civil_key = st.selectbox('Selecione o seu estado civil', estado_civil_key)
input_estado_civil_val = estado_civil_dict.get(input_estado_civil_key)

# Número de membros da família
st.write("""
### Família
""")
membros_familia_count = float(st.selectbox('Selecione quantos membros tem na sua família', [1,2,3,4,5,6,7,8,9,10,11,12]))

st.write("""
## Bens
""")

# Carro próprio
st.write("""
## Carro próprio
#""")
input_carro_proprio = st.radio('Você possui um automóvel?',['Sim','Não'], index=0)
input_carro_proprio_dict = {'Sim': 1, 'Não':0}
input_carro_proprio_val = input_carro_proprio_dict.get(input_carro_proprio)

# Casa própria
st.write("""
## Casa própria
#""")
input_casa_propria = st.radio('Você possui uma propriedade?',['Sim','Não'], index=0)
input_casa_propria_dict = {'Sim': 1, 'Não':0}
input_casa_propria_val = input_casa_propria_dict.get(input_casa_propria)

# Moradia
st.write("""
### Tipo de residência
""")
tipo_moradia_values = list(value_cnt_norm_cal(full_data,'Moradia').index)
tipo_moradia_key = ['Apartamento alugado', 'Casa/apartamento próprio',
                    'Habitação pública ', 'Mora com os pais', 
                    'Cooperativa habitacional', 'Apartamento comercial']
tipo_moradia_dict = dict(zip(tipo_moradia_key,tipo_moradia_values))
input_tipo_moradia_key = st.selectbox('Selecione o seu tipo de residência', tipo_moradia_key)
input_tipo_moradia_val = tipo_moradia_dict.get(input_tipo_moradia_key)

# Situação de emprego
st.write("""
### Categoria de renda
""")
categoria_renda_values = list(value_cnt_norm_cal(full_data,'Categoria_de_renda').index)
categoria_renda_key = ['Empregado','Associado comercial','Pensionista','Servidor público','Estudante']
categoria_renda_dict = dict(zip(categoria_renda_key,categoria_renda_values))
input_categoria_renda_key = st.selectbox('Selecione o seu status de trabalho atual', categoria_renda_key)
input_categoria_renda_val = categoria_renda_dict.get(input_categoria_renda_key)

# Ocupação
st.write("""
### Ocupação
""")
ocupacao_values = list(value_cnt_norm_cal(full_data,'Ocupacao').index)
ocupacao_key = ['Segurança', 'Vendas', 'Contabilidade', 'Construção Civil',
                'Gerência', 'Motorista', 'Equipe principal', 'Alta tecnologia',
                'Limpeza', 'Serviço privado', 'Cozinha', 'Baixa qualificação',
                'Medicina', 'Secretariado', 'Garçom', 'RH', 'Corretor imobiliário',
                'TI']
ocupacao_dict = dict(zip(ocupacao_key, ocupacao_values))
input_ocupacao_key = st.selectbox('Selecione qual é a sua ocupação atual', ocupacao_key)
input_ocupacao_val = ocupacao_dict.get(input_ocupacao_key)

# Tempo de experiência
st.write("""
### Experiência
""")
input_tempo_experiencia = float(st.slider('Selecione o seu tempo de experiência em anos', value=6, min_value=0, max_value=30, step=1))

# Rendimentos
st.write("""
### Rendimentos
""")
input_rendimentos = float(st.text_input('Digite o seu rendimento anual (em reais) e pressione ENTER para confirmar',0))

# Telefone trabalho
st.write("""
### Telefone corporativo
""")
input_telefone_trabalho = st.radio('Você tem um telefone corporativo?',['Sim','Não'], index=0)
telefone_trabalho_dict = {'Sim': 1, 'Não':0}
telefone_trabalho_val = telefone_trabalho_dict.get(input_telefone_trabalho)

# Telefone fixo
st.write("""
### Telefone fixo
""")
input_telefone = st.radio('Você tem um telefone fixo?',['Sim','Não'], index=0)
telefone_dict = {'Sim': 1, 'Não':0}
telefone_val = telefone_dict.get(input_telefone)

# Email 
st.write("""
### Email
""")
input_email = st.radio('Você tem um email?',['Sim','Não'], index=0)
email_dict = {'Sim': 1, 'Não':0}
email_val = email_dict.get(input_email)

# Lista de todas as variáveis: 
new_profile = [0, # ID_Cliente
                    input_carro_proprio_val, # Tem_carro
                    input_casa_propria_val, # Tem_Casa_Propria
                    telefone_trabalho_val, # Tem_telefone_trabalho
                    telefone_val, # Tem_telefone_fixo
                    email_val,  # Tem_email
                    membros_familia_count,  # Tamanho_Familia
                    input_rendimentos, # Rendimento_anual	
                    input_idade, # Idade
                    input_tempo_experiencia, # Anos_empregado
                    input_categoria_renda_val, # Categoria_de_renda
                    input_grau_escolaridade_val, # Grau_Escolaridade
                    input_estado_civil_val, # Estado_Civil	
                    input_tipo_moradia_val, # Moradia                                                  
                    input_ocupacao_val, # Ocupacao
                     0 # target (Mau)
                    ]

profile_to_predict_df = pd.DataFrame([new_profile],columns=train_df_copy.columns)

train_new_profile = pd.concat([train_df_copy,profile_to_predict_df],ignore_index=True)

train_new_profile = full_pipeline(train_new_profile)

profile_to_pred = train_new_profile.drop(['Mau'], axis=1)

# Predictions  
if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(profile_to_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons()
    else:
        st.error('### Infelizmente, não podemos liberar crédito para você agora!')
 
