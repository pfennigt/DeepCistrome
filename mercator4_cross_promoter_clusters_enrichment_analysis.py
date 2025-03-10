import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
pd.options.display.width=0

# Load mercator4-annotations of Arabidopsis thaliana proteome
mercator_data_all = pd.read_csv(filepath_or_buffer="data/mercator4_output.txt", sep="\t")
significant_enrichment_df = []

for lvl in [1, 2, 3]:
    print(f'Handling Level: {lvl} of mercator4 annotations')
    mercator_data = mercator_data_all.copy()
    mercator_data.dropna(inplace=True, how="any")
    mercator_data['IDENTIFIER'] = mercator_data.IDENTIFIER.str.upper()
    mercator_data['gene_id'] = [x.replace("'", '').split('.')[0] for x in mercator_data['IDENTIFIER']]
    mercator_data['annotation'] = [i.replace("'", '').split('.')[0] for i in mercator_data['NAME']]
    mercator_data = mercator_data[mercator_data['annotation'] != 'No Mercator4 annotation']
    mercator_data['NAME'] = ['.'.join(i.replace("'", '').split('.')[:lvl]) for i in mercator_data['NAME']]

    cluster_idx_to_name = pd.read_csv(filepath_or_buffer='data/cluster_ids_cluster_name.csv')
    cluster_idx_to_name_dict = {int(k):v for k,v in cluster_idx_to_name.values}

    predicted_clusters = pd.read_csv(filepath_or_buffer="data/prom_term_predictions.csv")
    predicted_clusters.rename({'Unnamed: 0': 'gene_id'}, axis='columns', inplace=True)
    predicted_clusters = predicted_clusters[['gene_id', 'cluster', 'cluster_col']]
    mercator_data = mercator_data.merge(predicted_clusters, how='inner', on='gene_id')
    mercator_data = mercator_data[mercator_data['cluster'] != 0]
    mercator_data.replace(cluster_idx_to_name_dict, inplace=True)
    mercator_data.drop_duplicates(subset=['gene_id', 'NAME', 'cluster'], inplace=True)
    print(mercator_data.head())

    network_data = []
    statistical_tests_df = []
    print(len(mercator_data['NAME'].unique()))
    for grp in mercator_data.groupby(['NAME']):
        enrich_dict = grp[1]['cluster'].value_counts().to_dict()
        assert grp[1].shape[0] == len(grp[1]['gene_id'].unique())
        tab, grp_name = grp[1], grp[0][0]
        for v in cluster_idx_to_name_dict.values():
            if v not in enrich_dict.keys():
                enrich_dict[v] = 0

            # generating contingency table
            c_and_g = tab[tab['cluster'] == v].shape[0]
            not_c_and_g = tab[tab['cluster'] != v].shape[0]
            c_and_not_g = mercator_data[(mercator_data['cluster'] == v) &
                                        (mercator_data['NAME'] != grp_name)].shape[0]
            not_c_and_not_g = mercator_data[(mercator_data['cluster'] != v) &
                                            (mercator_data['NAME'] != grp_name)].shape[0]
            # Performing fishers exact test
            statistic, pvalue = fisher_exact(np.array([[c_and_g, c_and_not_g],
                                                       [not_c_and_g, not_c_and_not_g]]),
                                             alternative='two-sided')
            statistical_tests_df.append([grp_name, v, pvalue])
        enrich_dict['NAME'] = grp_name
        network_data.append(enrich_dict)

    network_data = pd.DataFrame.from_records(data=network_data)
    network_data.drop(columns=['-'], inplace=True)
    statistical_tests_df = pd.DataFrame(statistical_tests_df, columns=['NAME', 'cluster', 'p.value'])
    print(network_data.head())

    # Melt table
    network_data = pd.melt(network_data,
                           id_vars='NAME',
                           value_vars=list(network_data.columns[1:]),
                           var_name='cluster',
                           value_name='Count')

    data_enrichments = network_data.merge(statistical_tests_df, how='inner', on=['NAME', 'cluster'])

    significant_mercator_enrichments = []
    for cluster in data_enrichments['cluster'].unique():
        data = data_enrichments[data_enrichments['cluster'] == cluster]
        rejected, pvalue_corrected = fdrcorrection(data['p.value'].tolist(), alpha=0.05)
        data['adj.p.value'] = pvalue_corrected
        data['FDR(-log10)'] = -1*np.log10(data['adj.p.value'])
        data['FDR'] = ['Sig.' if i < 0.05 else 'Not Sig.' for i in data['adj.p.value'].values]

        data.sort_values(by=['NAME', 'adj.p.value'], ascending=True, inplace=True)
        print(data.head(5))
        significant_mercator_enrichments.append(data)

    significant_mercator_enrichments = pd.concat(significant_mercator_enrichments)
    significant_mercator_enrichments = significant_mercator_enrichments[significant_mercator_enrichments['FDR'] == 'Sig.']
    print(significant_mercator_enrichments.head(20))
    print(significant_mercator_enrichments.shape)
    significant_enrichment_df.append(significant_mercator_enrichments)

significant_enrichment_df = pd.concat(significant_enrichment_df)
significant_enrichment_df['Count(log2)'] = np.log2(significant_enrichment_df['Count'])
significant_enrichment_df.sort_values(by=['NAME'], ascending=True, inplace=True)
print(significant_enrichment_df.head(50))
significant_enrichment_df.to_csv(path_or_buf='results/Supplementary_table_1_sheet_3.csv', index=False)
chart = alt.Chart(data=significant_enrichment_df
                  ).mark_point(filled=True
                               ).encode(y=alt.Y('NAME:N'),
                                        x=alt.X('cluster', sort=cluster_idx_to_name['cluster_name'].tolist()),
            color=alt.Color('FDR(-log10):Q').scale(scheme='purplered'), size='Count(log2):Q').configure_axis(labelLimit=1000)
chart.save(f"results/Figures/mercator_cluster_enrichment_analysis.svg")
