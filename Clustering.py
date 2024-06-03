import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
from scipy import stats

def check_df(dataframe, head=5):
    st.write("Shape:", dataframe.shape)
    st.write("Types:", dataframe.dtypes)
    st.write("Head:", dataframe.head(head))
    st.write("Tail:", dataframe.tail(head))
    st.write("NA:", dataframe.isnull().sum())
    st.write("Quantiles:", dataframe.describe().T)

def preprocess_data(data, handle_outliers=False, outlier_threshold=1.5,
                    handle_missing=False, missing_strategy='mean',
                    label_encode=False, drop_columns=None, scale_data=False):
    if drop_columns:
        data = data.drop(columns=drop_columns)
        st.write(f"Dropped columns: {drop_columns}, Shape: {data.shape}")

    if handle_missing:
        imputer = SimpleImputer(strategy=missing_strategy)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        st.write(f"Handled missing values with strategy '{missing_strategy}', Shape: {data.shape}")

    if handle_outliers:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        st.write(f"Handled outliers, Shape: {data.shape}")

    if label_encode:
        categorical_cols = data.select_dtypes(include=[object]).columns
        for col in categorical_cols:
            data[col] = LabelEncoder().fit_transform(data[col])
        st.write(f"Label encoded columns: {categorical_cols.tolist()}, Shape: {data.shape}")

    if scale_data:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        st.write(f"Scaled data, Shape: {data.shape}")

    return data

def kmeans_clustering(preprocessed_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preprocessed_data)
    return labels


def agglomerative_clustering(preprocessed_data, n_clusters, linkage):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agglomerative.fit_predict(preprocessed_data)
    return labels

def gaussian_mixture_model(preprocessed_data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(preprocessed_data)
    return labels

def mean_shift_clustering(preprocessed_data):
    ms = MeanShift()
    labels = ms.fit_predict(preprocessed_data)
    return labels





def visualize_clusters(data, labels, algorithm):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    num_clusters = len(set(labels))
    colors = alt.Color('Cluster:N', scale=alt.Scale(scheme='set1'))

    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x='PC1',
        y='PC2',
        color=colors,
        tooltip=['PC1', 'PC2', 'Cluster']
    ).properties(
        title=f'Cluster visualization for {algorithm}',
        width=800,
        height=600
    ).interactive()

    st.write(chart)

# Kullanımı

def evaluate_clusters(data, labels, method):
    if method == "Elbow Method":
        inertia_values = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            inertia_values.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        st.pyplot(plt)

        st.write("Elbow Method is not applicable for DBSCAN and Mean Shift algorithms.")

    elif method == "Silhouette Method":
        silhouette_avg = silhouette_score(data, labels)
        st.write(f"Silhouette Score: {silhouette_avg}")

    elif method == "Calinski-Harabasz Index":
        calinski_harabasz_index = calinski_harabasz_score(data, labels)
        st.write(f"Calinski-Harabasz Score: {calinski_harabasz_index}")

    elif method == "Davies-Bouldin Index":
        davies_bouldin_index = davies_bouldin_score(data, labels)
        st.write(f"Davies-Bouldin Score: {davies_bouldin_index}")


def main():
    st.title("Clustering and Analysis Project")
    st.sidebar.header("Upload your dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        check_df(data)

        st.sidebar.header("Preprocessing Options")
        handle_outliers = st.sidebar.checkbox("Handle outliers", value=False)
        handle_missing = st.sidebar.checkbox("Handle missing values", value=False)
        missing_strategy = st.sidebar.selectbox("Missing value strategy", ["mean", "median", "most_frequent"]) if handle_missing else None
        label_encode = st.sidebar.checkbox("Label encode categorical data", value=False)
        drop_columns = st.sidebar.multiselect("Drop columns", options=data.columns.tolist())
        scale_data = st.sidebar.checkbox("Scale data", value=False)

        if st.sidebar.button("Preprocess Data"):
            st.session_state.preprocessed_data = preprocess_data(data, handle_outliers, 1.5, handle_missing, missing_strategy, label_encode, drop_columns, scale_data)
            st.write("Preprocessed Data:")
            st.write(st.session_state.preprocessed_data)

        if 'preprocessed_data' in st.session_state:
            preprocessed_data = st.session_state.preprocessed_data

            st.sidebar.header("Clustering Options")
            clustering_algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "Agglomerative Clustering", "Gaussian Mixture", "Mean Shift"])

            if clustering_algorithm == "KMeans":
                n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3)
                labels = kmeans_clustering(preprocessed_data, n_clusters)

                # Custom scoring function for DBSCAN

            elif clustering_algorithm == "Agglomerative Clustering":
                n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3)
                linkage = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"])
                labels = agglomerative_clustering(preprocessed_data, n_clusters, linkage)
            elif clustering_algorithm == "Gaussian Mixture":
                n_components = st.sidebar.slider("Number of components", min_value=2, max_value=10, value=3)
                labels = gaussian_mixture_model(preprocessed_data, n_components)
            elif clustering_algorithm == "Mean Shift":
                labels = mean_shift_clustering(preprocessed_data)

            st.session_state.labels = labels
            st.write("Cluster Labels:", labels)
            labels = labels.flatten()

            # Verilerinizi ve etiketlerinizi yazdırın
            #st.write(labels.shape)
            #st.write(data.shape)

            # Verileri kopyalayın ve etiketleri DataFrame'e ekleyin
            clustered_data = data.copy()
            clustered_data['cluster'] = labels

            st.dataframe(clustered_data)
            visualize_clusters(preprocessed_data, labels, clustering_algorithm)

            st.sidebar.header("Evaluation Options")
            evaluation_method = st.sidebar.selectbox("Select Evaluation Method", ["Elbow Method", "Silhouette Method", "Calinski-Harabasz Index", "Davies-Bouldin Index"])
            if st.sidebar.button("Evaluate Clusters"):
                evaluate_clusters(preprocessed_data, labels, evaluation_method)
            st.subheader("Further Analysis")
            st.sidebar.header("Visualization")
            group_column = st.sidebar.selectbox("Select Group Column", options=clustered_data.columns.tolist())
            if st.button("Show Plot"):


                # Cluster ve gerçek değerler arasındaki ilişkiyi gösteren bir görselleştirme
                clustered_data['Cluster'] = clustered_data['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})

                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=clustered_data, x='diagnosis', y='Cluster', hue='Cluster')
                plt.title('Cluster vs. Diagnosis')
                plt.xlabel('Diagnosis')
                plt.ylabel('Cluster')
                st.pyplot(plt)
                # Önce one-hot encoded sınıfların dağılımını görelim
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='id', y='cluster', data=clustered_data, label='Cluster', alpha=0.7)

                # Sonra gerçek sınıfların dağılımını görelim

                # Verilerinizi ve etiketlerinizi yazdırın
                st.write(labels.shape)
                st.write(data.shape)
                # Çift yoğunluk grafiği oluştur
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(data=data, x='id', hue='diagnosis', fill=True, common_norm=False, palette='husl', ax=ax)
                plt.title('Cluster and Diagnosis Distribution')
                plt.xlabel('ID')
                plt.ylabel('Density')
                st.pyplot(fig)
                # Create a figure and axis
                fig, ax = plt.subplots()

                # Plotting the 'cluster' column
                cluster_counts = clustered_data['cluster'].value_counts()
                sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, label='Cluster', color='blue')

                # Plotting the 'diagnosis' column
                diagnosis_counts = clustered_data['diagnosis'].value_counts()
                sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, ax=ax, label='Diagnosis',
                            color='green')

                # Adding labels and title
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Cluster and Diagnosis')

                # Displaying the legend
                ax.legend()

                # Showing the plot
                st.pyplot(fig)


                # Canlı renkler için bir renk paleti oluşturalım
                colors = ['#FF6347', '#7B68EE', '#32CD32', '#FFD700', '#BA55D3', '#FFA500', '#1E90FF', '#FF69B4',
                          '#00FFFF', '#8A2BE2']

                # cluster ve diagnosis sütunlarını aynı tür yapalım
                clustered_data['cluster'] = clustered_data['cluster'].astype(str)
                clustered_data['diagnosis'] = clustered_data['diagnosis'].astype(str)

                # Sınıfları gruplayarak örnek sayısını hesaplayalım
                counts_cluster = clustered_data['cluster'].value_counts()
                counts_diagnosis = clustered_data['diagnosis'].value_counts()

                # Bar grafiklerini oluşturalım
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                # Cluster sütunu
                sns.barplot(x=counts_cluster.index, y=counts_cluster.values, ax=ax[0], palette=colors)
                ax[0].set_title('Cluster Distribution', fontsize=16)
                ax[0].set_xlabel('Cluster', fontsize=14)
                ax[0].set_ylabel('Count', fontsize=14)
                for i, v in enumerate(counts_cluster.values):
                    ax[0].text(i, v + 0.1, str(v), ha='center', fontsize=12)

                # Diagnosis sütunu
                sns.barplot(x=counts_diagnosis.index, y=counts_diagnosis.values, ax=ax[1], palette=colors)
                ax[1].set_title('Diagnosis Distribution', fontsize=16)
                ax[1].set_xlabel('Diagnosis', fontsize=14)
                ax[1].set_ylabel('Count', fontsize=14)
                for i, v in enumerate(counts_diagnosis.values):
                    ax[1].text(i, v + 0.1, str(v), ha='center', fontsize=12)

                plt.tight_layout()
                st.pyplot(fig)


if __name__ == "__main__":
    main()

