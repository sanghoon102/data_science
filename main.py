import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
all_batter_stat_filtered=pd.read_csv('all_batter_stat_filtered.csv')
all_injure=pd.read_csv('all_injure.csv')
all_player=pd.read_csv('all_player.csv')
all_salary_over5000=pd.read_csv('all_salary_over5000.csv')
relief_pitcher_stat=pd.read_csv('relief_pitcher_stat.csv')
starting_pitcher_stat=pd.read_csv('starting_pitcher_stat.csv')
all_batter_stat=pd.read_csv('all_batter_stat.csv')
all_pitcher_stat=pd.read_csv('all_pitcher_stat.csv')
refined_batter_stat=pd.read_csv('refined_batter_stat.csv')
refined_pitcher_stat=pd.read_csv('refined_pitcher_stat.csv')
# Example DataFrame creation (replace with your actual DataFrame)
all_salary_over5000.rename(columns={'연봉': 'salary'}, inplace=True)
all_batter_stat_filtered.rename(columns={'연봉': 'salary','타율':'avg','출루':'obp','장타':'slg','홈런':'HR'}, inplace=True)
starting_pitcher_stat.rename(columns={'연봉': 'salary','이닝':'inning'}, inplace=True)
relief_pitcher_stat.rename(columns={'연봉': 'salary','이닝':'inning'}, inplace=True)
all_injure.rename(columns={'기간':'injured date','연봉 차액':'salary up and down'},inplace=True)
all_player.rename(columns={'손잡이':'handedness','연봉':'salary','나이':'age'},inplace=True)
all_player['handedness'] = all_player['handedness'].replace({'좌': 'Left', '우': 'Right'})
# Streamlit app title
st.title('KBO 연봉 분석과 예측')

# Sidebar for data selection
st.sidebar.header('')
options = ['앱 개요','war 과 연봉', '타자 성적과 연봉', '선발 투수 성적과 연봉', '불펜 투수 성적과 연봉', '부상이력과 연봉 증감', '나이와 연봉', '손잡이와 연봉', '연봉 예측']
selected_option = st.sidebar.selectbox('항목을 선택하세요', options)

# Data visualization
st.header(selected_option)
if selected_option=='앱 개요':
    st.write('**war 과 연봉**')
    st.write('''WAR (Wins Above Replacement)과 연봉의 관계를 시각화하고 분석합니다.
    WAR과 연봉 사이의 상관 계수를 계산하고, 이를 해석합니다.''')
    st.write('')
    st.write('**타자 성적과 연봉**')
    st.write('''타자의 성적 지표(타율, 출루율, 장타율, 홈런, dWAR)와 연봉의 관계를 시각화하고 분석합니다.
    각 지표와 연봉 사이의 상관 계수를 계산하고, 이를 해석합니다.''')
    st.write('')
    st.write('**선발 투수 성적과 연봉**')
    st.write('''선발 투수의 성적 지표(ERA, FIP, WHIP, 이닝)와 연봉의 관계를 시각화하고 분석합니다.
    각 지표와 연봉 사이의 상관 계수를 계산하고, 이를 해석합니다.''')
    st.write('')
    st.write('**불펜 투수 성적과 연봉**')
    st.write('''불펜 투수의 성적 지표(ERA, FIP, WHIP, 이닝)와 연봉의 관계를 시각화하고 분석합니다.
     각 지표와 연봉 사이의 상관 계수를 계산하고, 이를 해석합니다.''')
    st.write('')
    st.write('**부상이력과 연봉 증감**')
    st.write('''부상 일수와 연봉 차액의 관계를 분석합니다.
    부상 일수별 연봉 차액의 평균을 계산하고 분석합니다.''')
    st.write('')
    st.write('**나이와 연봉**')
    st.write('''선수의 나이와 연봉의 관계를 시각화하고 분석합니다.
    각 나이 구간별 연봉의 평균을 계산하고, 이를 시각화합니다.''')
    st.write('')
    st.write('**연봉 예측**')
    st.write('''타자와 투수의 성적 데이터를 기반으로 연봉을 예측하는 모델을 제공합니다.
    사용자 입력을 통해 새로운 데이터를 예측하고, 결과를 출력합니다.''')
elif selected_option == 'war 과 연봉':
    plt.figure(figsize=(10, 6))
    sns.regplot(data=all_salary_over5000, x='war', y='salary', line_kws={'color': 'red'})
    plt.title('WAR vs Salary with Trend Line')
    st.pyplot(plt)
    correlation_coefficient = all_salary_over5000['war'].corr(all_salary_over5000['salary'])
    st.write(f'war과 연봉의 상관 계수: {correlation_coefficient}')

    st.write("""
    WAR과 연봉의 상관계수는 0.45로 절댓값이 거의 0.5인데 이는 통계적으로 상관이 있다고 지지받을 수 있는 수치이다. 
    수치가 1에 가깝지 않는 것은 선수의 연봉이 선수의 인지도, 팀의 재정 상태, 포지션, 나이 등 다양한 요인에 영향을 받기 때문입니다.

    그래도 선수들의 승리기여도(WAR)가 어느 정도 연봉과 상관이 있음을 보여주는 수치입니다.
    """)





elif selected_option == '타자 성적과 연봉':

    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["타율", "출루율", "장타율", "홈런", "dWAR(수비 기여도)",'분석'])

    with tab1:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=all_batter_stat_filtered, x='avg', y='salary', line_kws={'color': 'red'})
        plt.title('Batting Average vs Salary with Trend Line')
        st.pyplot(plt)

        correlation_coefficient_avg = all_batter_stat_filtered['avg'].corr(all_batter_stat_filtered['salary'])
        st.write(f'타율과 연봉의 상관 계수: {correlation_coefficient_avg}')

    with tab2:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=all_batter_stat_filtered, x='obp', y='salary', line_kws={'color': 'red'})
        plt.title('On Base Property vs Salary with Trend Line')
        st.pyplot(plt)

        correlation_coefficient_obp = all_batter_stat_filtered['obp'].corr(all_batter_stat_filtered['salary'])
        st.write(f'출루율과 연봉의 상관 계수: {correlation_coefficient_obp}')

    with tab3:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=all_batter_stat_filtered, x='slg', y='salary', line_kws={'color': 'red'})
        plt.title('Slugging Percentage vs Salary with Trend Line')
        st.pyplot(plt)

        correlation_coefficient_slg = all_batter_stat_filtered['slg'].corr(all_batter_stat_filtered['salary'])
        st.write(f'장타율과 연봉의 상관 계수: {correlation_coefficient_slg}')

    with tab4:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=all_batter_stat_filtered, x='HR', y='salary', line_kws={'color': 'red'})
        plt.title('Home run vs Salary with Trend Line')
        st.pyplot(plt)

        correlation_coefficient_HR = all_batter_stat_filtered['HR'].corr(all_batter_stat_filtered['salary'])
        st.write(f'홈런과 연봉의 상관 계수: {correlation_coefficient_HR}')

    with tab5:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=all_batter_stat_filtered, x='dWAR', y='salary', line_kws={'color': 'red'})
        plt.title('Defending WAR vs Salary with Trend Line')
        st.pyplot(plt)

        correlation_coefficient_dWAR = all_batter_stat_filtered['dWAR'].corr(all_batter_stat_filtered['salary'])
        st.write(f'수비 기여도와 연봉의 상관 계수: {correlation_coefficient_dWAR}')
    
    with tab6:

        st.write(f'타율과 연봉의 상관 계수: {correlation_coefficient_avg}')
        st.write(f'출루율과 연봉의 상관 계수: {correlation_coefficient_obp}')
        st.write(f'장타율과 연봉의 상관 계수: {correlation_coefficient_slg}')
        st.write(f'홈런과 연봉의 상관 계수: {correlation_coefficient_HR}')
        st.write(f'dWAR과 연봉의 상관 계수: {correlation_coefficient_dWAR}')
        st.write('')
        st.write('타자의 성적은 타율이나 출루율보단 장타율이나 홈런이 상관계수가 더 큰것으로 나타났다. ')
        st.write('이는 구단들이 안타와 출루보단 팀의 득점력과 직접 관련이 있는 장타율과 홈런에 더 많은 비중을 두고 있다고 생각할 수 있다.')
        st.write('또한 홈런과 장타가 많은 선수는 인지도 면에서도 높기 때문에 연봉을 많이 받는다고 생각할 수 있다.')
        st.write('의외로 연봉과 dWAR은 음의 상관관게를 보였는데 이는 득점 생산력이 좋아 연봉이 높고 장타율과 홈런이 많은 선수는 주로 1루수나 지명타자와 같은 포지션에 배치되기 때문에 수비에 대한 기여는 낮기 때문이라고 생각할 수 있다.')




elif selected_option == '선발 투수 성적과 연봉':
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["ERA(평균자책점)", "FIP(수비 무관 평균자책점)", "WHIP(이닝 당 출루 허용률)", "이닝",'분석'])

    with tab1:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=starting_pitcher_stat, x='ERA', y='salary', line_kws={'color': 'red'})
        plt.title('ERA vs Salary with Trend Line')
        st.pyplot(plt)

        s_correlation_coefficient_ERA = starting_pitcher_stat['ERA'].corr(starting_pitcher_stat['salary'])
        st.write(f'ERA와 연봉의 상관 계수: {s_correlation_coefficient_ERA}')

    with tab2:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=starting_pitcher_stat, x='FIP', y='salary', line_kws={'color': 'red'})
        plt.title('FIP vs Salary with Trend Line')
        st.pyplot(plt)

        s_correlation_coefficient_FIP = starting_pitcher_stat['FIP'].corr(starting_pitcher_stat['salary'])
        st.write(f'FIP와 연봉의 상관 계수: {s_correlation_coefficient_FIP}')

    with tab3:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=starting_pitcher_stat, x='WHIP', y='salary', line_kws={'color': 'red'})
        plt.title('WHIP vs Salary with Trend Line')
        st.pyplot(plt)

        s_correlation_coefficient_WHIP = starting_pitcher_stat['WHIP'].corr(starting_pitcher_stat['salary'])
        st.write(f'WHIP와 연봉의 상관 계수: {s_correlation_coefficient_WHIP}')

    with tab4:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=starting_pitcher_stat, x='inning', y='salary', line_kws={'color': 'red'})
        plt.title('inning vs Salary with Trend Line')
        st.pyplot(plt)

        s_correlation_coefficient_inning = starting_pitcher_stat['inning'].corr(starting_pitcher_stat['salary'])
        st.write(f'이닝과 연봉의 상관 계수: {s_correlation_coefficient_inning}')
    with tab5:
        st.write(f'ERA와 연봉의 상관 계수: {s_correlation_coefficient_ERA}')
        st.write(f'FIP와 연봉의 상관 계수: {s_correlation_coefficient_FIP}')
        st.write(f'WHIP와 연봉의 상관 계수: {s_correlation_coefficient_WHIP}')
        st.write(f'이닝과 연봉의 상관 계수: {s_correlation_coefficient_inning}')
        st.write('')
        st.write('''ERA(평균 자책점), FIP(수비 무관 평균자책점), WHIP(이닝 당 출루 허용률) 이 세 지표는 모두 -0.3정도의 조금 약한 음의 상관관계를 보인다. 
        이는 이 지표들이 낮을수록(즉, 투수의 경기력이 뛰어날수록) 연봉이 높아지는 경향을 나타낸다. 
        그러나 그렇게 강한 상관관계는 아니라는 것을 알 수 있다.

        
        ''')
        st.write('''이닝은 연봉과 0.5정도의 양의 상관관계를 보여준다. 
        이는 많은 이닝을 던지는 투수가 더 높은 연봉을 받는 경향이 있음을 나타낸다. 
        다른 지표보다는 상관계수의 절댓값이 높은것을 통해 구단들은 이닝을 많이 던진(즉, 꾸준하게 부상없이 피칭을 한)선수들에게 연봉을 많이 준다는것을 알 수 있다.''')


elif selected_option == '불펜 투수 성적과 연봉':
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["ERA(평균자책점)", "FIP(수비 무관 평균자책점)", "WHIP(이닝 당 출루 허용률)", "이닝",'분석'])

    with tab1:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=relief_pitcher_stat, x='ERA', y='salary', line_kws={'color': 'red'})
        plt.title('ERA vs Salary with Trend Line')
        st.pyplot(plt)

        r_correlation_coefficient_ERA =relief_pitcher_stat['ERA'].corr(relief_pitcher_stat['salary'])
        st.write(f'ERA와 연봉의 상관 계수: {r_correlation_coefficient_ERA}')

    with tab2:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=relief_pitcher_stat, x='FIP', y='salary', line_kws={'color': 'red'})
        plt.title('FIP vs Salary with Trend Line')
        st.pyplot(plt)

        r_correlation_coefficient_FIP = relief_pitcher_stat['FIP'].corr(relief_pitcher_stat['salary'])
        st.write(f'FIP와 연봉의 상관 계수: {r_correlation_coefficient_FIP}')

    with tab3:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=relief_pitcher_stat, x='WHIP', y='salary', line_kws={'color': 'red'})
        plt.title('WHIP vs Salary with Trend Line')
        st.pyplot(plt)

        r_correlation_coefficient_WHIP = relief_pitcher_stat['WHIP'].corr(relief_pitcher_stat['salary'])
        st.write(f'WHIP와 연봉의 상관 계수: {r_correlation_coefficient_WHIP}')

    with tab4:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=relief_pitcher_stat, x='inning', y='salary', line_kws={'color': 'red'})
        plt.title('inning vs Salary with Trend Line')
        st.pyplot(plt)

        r_correlation_coefficient_inning = relief_pitcher_stat['inning'].corr(relief_pitcher_stat['salary'])
        st.write(f'이닝과 연봉의 상관 계수: {r_correlation_coefficient_inning}')

    with tab5:
        st.write(f'ERA와 연봉의 상관 계수: {r_correlation_coefficient_ERA}')
        st.write(f'FIP와 연봉의 상관 계수: {r_correlation_coefficient_FIP}')
        st.write(f'WHIP와 연봉의 상관 계수: {r_correlation_coefficient_WHIP}')
        st.write(f'이닝과 연봉의 상관 계수: {r_correlation_coefficient_inning}')
        st.write('')
        st.write('''이닝과 연봉의 상관관계 수치는 -0.005로써 상관관계가 없다고 볼 수 있다. 이는 불펜 투수의 경우, 선발 투수와 달리 많은 이닝을 던지는 것이 연봉 결정에 큰 영향을 미치지 않는것으로 해석된다.''')
        st.write('''나머지 ERA, WHIP, FIP는 상관계수가 모두 -0.2정도로 약한 음의 상관관계를 보인다. 이로써 이 성적 지표들이 낮을수록(즉, 투수의 성적이 좋을수록) 연봉이 높아지는 경향이 있지만 그 영향력은 상대적으로 약하다는것을 의미한다.''')
        st.write('''불펜 투수의 경우, 특정 상황에서의 투구 능력, 경기 상황에 대한 기여도 등 다른 요인들이 연봉 결정에 더 큰 영향을 미칠 수 있기 때문이라고 해석된다.''')




elif selected_option == '부상이력과 연봉 증감':
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=all_injure, x='injured date', y='salary up and down')
    plt.title('Injury Date vs Salary up and down')
    st.pyplot(plt)

    injury_days = [10, 15, 20, 25, 30]
    injury_salary_mean = all_injure[all_injure['injured date'].isin(injury_days)].groupby('injured date')['salary up and down'].mean().reset_index()

    st.subheader('부상 일수별 연봉 차액 평균')
    for _, row in injury_salary_mean.iterrows():
        st.write(f"{int(row['injured date'])}일: {row['salary up and down']:,.0f}만원")

    st.write('''부상일수가 10일인 선수들은 연봉 차액이 양수가 나왔다. 
            이는 10일의 부상은 부상 때문에 연봉이 딱히 감소되지 않고 선수의 성적이 더 중요한것으로 판단된다.

            ''')
    st.write('부상일수가 15일~30일 사이의 선수들은 평균적으로 연봉이 삭감되었다.')






elif selected_option == '나이와 연봉':
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=all_player, x='age', y='salary')
    plt.title('Age vs Salary')
    st.pyplot(plt)

    bins = range(20, 45, 4)
    labels = ['20-24', '24-28', '28-32', '32-36', '36-40', '40-44']
    all_player['나이 구간'] = pd.cut(all_player['age'], bins=bins,labels=labels)

    # 각 나이 구간별 연봉의 평균 구하기
    age_salary_mean = all_player.groupby('나이 구간')['salary'].mean().round(0).reset_index()

    # 각 구간의 연봉 평균값 계산

    st.subheader('연령대별 연봉 평균')
    st.dataframe(age_salary_mean)

    st.write('''20~24세 사이의 젊은 선수들 중에서는 상대적으로 낮은 연봉을 받는 경우가 많습니다. 
    이는 경험 부족, 신인 계약 등의 이유로 설명될 수 있습니다.''')

    st.write('''24~28세 사이의 선수들의 연봉도 비교적 낮습니다. 
    이 시기에는 아직 기량을 완전히 발휘하지 못했거나 팀 내에서 주전 자리를 완전히 확보하지 못한 경우가 많을 수 있습니다.''')

    st.write('''28~32세 사이의 선수들은 연봉이 증가하는 경향이 있습니다. 
    선수들이 기량을 완전히 발휘하고, 팀의 주전으로 자리 잡을 가능성이 높아지기 때문입니다.''')

    st.write('''32~36세 사이의 선수 들연봉이 가장 높습니다. 
    이 시기의 선수들은 경험이 풍부하고, 경기력이 최고조에 달할 가능성이 높습니다. 따라서 팀에서 중요한 역할을 하며 높은 연봉을 받게 됩니다.''')

    st.write('''36~40세 사이의 선수들은 연봉이 약간 감소하는 경향이 있지만 여전히 높은 수준을 유지합니다. 
    이 시기의 선수들은 여전히 기량을 유지하고 있지만, 나이로 인해 체력이나 경기력에서 점차 하락세를 보일 수 있습니다.''')

    st.write('''40~44세 사이의 선수들은 연봉이 크게 감소합니다. 
    이 시기의 선수들은 대부분 은퇴를 앞두고 있거나, 기량이 하락한 경우가 많아 연봉이 낮아지는 경향을 보입니다.''')



elif selected_option == '손잡이와 연봉':
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=all_player, x='handedness', y='salary')
    plt.title('Handedness vs Salary')
    st.pyplot(plt)

    handedness_salary_mean = all_player.groupby('handedness')['salary'].mean()

    # 손잡이별 WAR 평균값 계산
    handedness_war_mean = all_player.groupby('handedness')['war'].mean()

    st.subheader('손잡이별 연봉 및 WAR 평균')
    for handedness in handedness_salary_mean.index:
        if handedness=='우':
            st.write(f"오른손 잡이 연봉 평균: {handedness_salary_mean[handedness]:,.0f}만원")
            st.write(f"오른손 잡이 WAR 평균: {handedness_war_mean[handedness]:.2f}")
        else:
            st.write(f"왼손 잡이 연봉 평균: {handedness_salary_mean[handedness]:,.0f}만원")
            st.write(f"왼손 잡이 WAR 평균: {handedness_war_mean[handedness]:.2f}")
    st.write('')
    st.write('''왼손 잡이 선수들의 평균 연봉은 21,141만원으로, 오른손잡이 선수들의 평균 연봉인 20,554만원보다 약간 더 높습니다.''')
    st.write('오른손 잡이 선수들의 평균 WAR은 1.395로, 왼손잡이 선수들의 평균 WAR인 1.291보다 높습니다.')
    st.write('''왼손 잡이 선수들은 평균적으로 오른손잡이 선수들보다 약간 더 높은 연봉을 받고 있지만, WAR은 오른손잡이 선수들이 더 높습니다. 
            이러한 결과는 연봉이 꼭 WAR과 정비례하지 않는다는 것을 보여줍니다. 
            연봉은 선수의 시장 가치, 희소성, 경험, 계약 시점의 상황 등 여러 요인에 의해 결정될 수 있습니다.''')
    st.write('오른손잡이 선수들이 더 높은 WAR을 보이지만, 연봉에서는 왼손잡이 선수들이 약간 더 높다는 것은 왼손잡이 선수들이 특정 포지션에서 더 큰 전략적 가치를 가지고 있을 수 있음을 시사합니다.')








elif selected_option == '연봉 예측':
    tab1, tab2 = st.tabs(["batter salary predict", "pitcher salary predict"])

    # 타자 연봉 예측
    with tab1:
        st.header("batter salary predict")
        # Prepare data
        X = refined_batter_stat[['타율', '출루', '장타', '홈런', 'war', 'dWAR']]  # Example using batter stats
        y = refined_batter_stat['연봉']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        batter_model = LinearRegression()
        batter_model.fit(X_train, y_train)

        y_pred_best = batter_model.predict(X_test)
        mse_best = mean_squared_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)

        # Predict and evaluate
        y_pred_best = batter_model.predict(X_test)
        mse_best = mean_squared_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)

        st.write(f"Model Mean Squared Error: {mse_best}")
        st.write(f"Model accuracy: {r2_best}")

        # New data input for prediction
        st.subheader('New Data Input')
        batting_avg = st.number_input('타율', 0.0, 1.0, step=0.01)
        on_base_percentage = st.number_input('출루율', 0.0, 1.0, step=0.01)
        slugging_percentage = st.number_input('장타율', 0.0, 1.0, step=0.01)
        home_runs = st.number_input('홈런', 0, 100, step=1)
        war = st.number_input('war', -9.0, 15.0, step=0.1)
        dwar = st.number_input('dWAR', -10.0, 10.0, step=0.1)

        new_data = pd.DataFrame({
            '타율': [batting_avg],
            '출루': [on_base_percentage],
            '장타': [slugging_percentage],
            '홈런': [home_runs],
            'war': [war],
            'dWAR': [dwar]
        })

        if st.button('타자 연봉 예측하기'):
            predicted_salary = batter_model.predict(new_data)
            if predicted_salary<2700:
                st.write(f"예측된 연봉: 2700(만원)")
            else:
                st.write(f"예측된 연봉: {predicted_salary[0]:,.0f} (만원)")

    # 투수 연봉 예측
    with tab2:
        st.header("pitcher salary predict")
        # Prepare data
        X = refined_pitcher_stat[['이닝', 'ERA', 'WHIP', 'FIP','war']]  # Example using batter stats
        y = refined_pitcher_stat['연봉']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        pitcher_model = LinearRegression()
        pitcher_model.fit(X_train, y_train)

        y_pred_best = pitcher_model.predict(X_test)
        mse_best = mean_squared_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)

        # Predict and evaluate
        y_pred_best = pitcher_model.predict(X_test)
        mse_best = mean_squared_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)

        st.write(f"Model Mean Squared Error: {mse_best}")
        st.write(f"Model accuracy: {r2_best}")

        # New data input for prediction
        st.subheader('New Data Input')
        inning = st.number_input('이닝', 0.0, 200.0, step=0.1)
        ERA = st.number_input('ERA(평균 자책점)', 0.0, 10.0, step=0.01)
        WHIP = st.number_input('WHIP(이닝 당 출루 허용률)', 0.0, 10.0, step=0.01)
        FIP = st.number_input('FIP(수비 무관 평균 자책점)', 0.0, 10.0, step=0.01)
        war = st.number_input('war', -10.0, 15.0, step=0.1)


        new_data = pd.DataFrame({
            '이닝': [inning],
            'ERA': [on_base_percentage],
            'WHIP': [slugging_percentage],
            'FIP': [home_runs],
            'war': [war]
        })

        if st.button('투수 연봉 예측하기'):
            predicted_salary = pitcher_model.predict(new_data)
            if predicted_salary<2700:
                st.write(f"예측된 연봉: 2700(만원)")
            else:
                st.write(f"예측된 연봉: {predicted_salary[0]:,.0f} (만원)")