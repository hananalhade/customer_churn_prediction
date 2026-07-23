import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. التنسيق - بنفسجي اغمق
st.set_page_config(page_title="تصميم نموذج للتعلم الآلي للتنبؤ بمغادرة العملاء في قطاع الاتصالات", layout="wide", page_icon="📡")
st.markdown("""
<style>
.stButton>button {
   background: linear-gradient(90deg, #9333EA, #6B21A8);
   color: white; border-radius: 15px; width: 100%; height: 3.5em;
   font-size:17px; font-weight:bold; border: none;
 }
.stButton>button:hover {
   background: linear-gradient(90deg, #6B21A8, #581C87);
 }
    h1, h2, h3 {color: #6B21A8;}
 .main {background-color: #F3E8FF;}
.metric-card {background: #E9D5FF; padding: 20px; border-radius: 15px; text-align: center;}
 div[data-testid="stMetric"] {background: #E9D5FF; padding: 15px; border-radius: 10px; border: 2px solid #C4B5FD;}
</style>
""", unsafe_allow_html=True)

# 2. قاعدة البيانات من الملف
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv('customers_data.csv')

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# 3. تسجيل الدخول
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("📡 تصميم نموذج للتعلم الآلي للتنبؤ بمغادرة العملاء في قطاع الاتصالات")
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.info("**بيانات الدخول:** المستخدم: `admin` | كلمة السر: `1234`")
        user = st.text_input("👤 اسم المستخدم")
        password = st.text_input("🔒 كلمة السر", type="password")
        if st.button("دخول النظام", use_container_width=True):
            if user == "admin" and password == "1234":
                st.session_state.logged_in = True; st.rerun()
            else: st.error("❌ خطأ في البيانات")
    st.stop()

df = st.session_state.df
menu = st.sidebar.selectbox("📋 القائمة الرئيسية",
    ["الرئيسية","ادارة العملاء","التقارير والرسوم","تدريب النموذج","تطبيق التوقع","تحليل اداء العملاء","تصدير التقارير","الاعدادات","تسجيل الخروج"])

# 4. الصفحات - كاملة
if menu == "الرئيسية":
    st.title("🏠 لوحة التحكم الرئيسية")
    st.write("مرحبا بك في نظام تصميم نموذج للتعلم الآلي للتنبؤ بمغادرة العملاء في قطاع الاتصالات")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👥 اجمالي العملاء", len(df))
    c2.metric("📉 عملاء معرضين للمغادرة", int(df['Churn'].sum()))
    c3.metric("✅ عملاء مستمرين", len(df)-int(df['Churn'].sum()))
    c4.metric("📊 نسبة الـ Churn", f"{df['Churn'].mean()*100:.1f}%")
    st.markdown("---")
    st.subheader("📊 رسم بياني: توزيع انواع العقود")
    fig = px.bar(df['Contract'].value_counts(), title='', color_discrete_sequence=['#9333EA'])
    st.plotly_chart(fig, use_container_width=True)

elif menu == "ادارة العملاء":
    st.title("👥 ادارة قاعدة البيانات")
    c1,c2,c3 = st.columns(3)
    if c1.button("➕ اضافة عميل جديد"):
    new_id = df['CustomerID'].max() + 1

    new_row = pd.DataFrame([[new_id,30,12,70,840,'','شهر بشهر']], columns=df.columns)
    st.session_state.df = pd.concat([df, new_row], ignore_index=True)
        st.success(f"✅ تمت اضافة العميل رقم {new_id}"); st.rerun()
    if c2.button("✏️ زيادة عمر اخر عميل"):
        st.session_state.df.loc[st.session_state.df.index[-1], 'Age'] += 1
        st.success("✅ تم التعديل"); st.rerun()
    if c3.button("🗑️ حذف اخر عميل"):
        if len(df)>1:
            st.session_state.df = df.iloc[:-1]
            st.warning("🗑️ تم الحذف"); st.rerun()
    st.dataframe(st.session_state.df, use_container_width=True, height=400)

elif menu == "التقارير والرسوم":
    st.title("📊 التقارير والرسوم البيانية")
    col1,col2 = st.columns(2)
    fig1 = px.histogram(df, x='Age', title='توزيع الاعمار', color_discrete_sequence=['#9333EA'])
    col1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x='Tenure', y='MonthlyCharges', title='المدة مقابل الفاتورة', color_discrete_sequence=['#6B21A8'])
    col2.plotly_chart(fig2, use_container_width=True)

elif menu == "تدريب النموذج":
    st.title("🧠 تدريب نموذج التعلم الآلي")
    if st.button("🚀 ابدأ تدريب النموذج", use_container_width=True):
        with st.spinner("جاري التدريب..."):
            df_train = df.copy()
            df_train['Contract'] = df_train['Contract'].map({'شهر بشهر':0,'سنة':1,'سنتين':2})
            X = df_train[['Age','Tenure','MonthlyCharges','Contract']]; y = df_train['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.session_state.model_trained = True; st.session_state.model = model
        st.success("✅ تم تدريب النموذج بنجاح!")
        st.metric("📈 دقة النموذج", f"{acc*100:.2f}%")
        st.progress(acc)

elif menu == "تطبيق التوقع":
    st.title("🔮 تطبيق توقع مغادرة العملاء")
    if not st.session_state.model_trained:
        st.warning("⚠️ الرجاء تدريب النموذج اولا")
    else:
        col1,col2 = st.columns(2)
        age = col1.number_input("العمر", 18, 100, 30)
        tenure = col2.slider("المدة بالشهور", 0, 72, 12)
        contract = st.selectbox("نوع العقد", ["شهر بشهر","سنة","سنتين"])
        charges = st.number_input("الفاتورة الشهرية $", 0.0, 200.0, 70.0)
        if st.button("🔍 توقع الان", use_container_width=True):
            contract_map = {'شهر بشهر':0,'سنة':1,'سنتين':2}
            input_data = [[age, tenure, charges, contract_map[contract]]]
            proba = st.session_state.model.predict_proba(input_data)[0][1] * 100
            if proba > 50: st.error(f"⚠️ تحذير: معرض للمغادرة بنسبة {proba:.1f}%")
            else: st.success(f"✅ مستمر بنسبة {100-proba:.1f}%")
            st.progress(proba/100)

elif menu == "تحليل اداء العملاء":
    st.title("📈 تحليل اداء العملاء")
    df_temp = df.copy()
    df_temp['الحالة'] = df_temp['Churn'].map({0:'مستمر',1:'هارب'})
    fig = px.box(df_temp, x='الحالة', y='MonthlyCharges', title='متوسط الفاتورة حسب الحالة', color='الحالة', color_discrete_sequence=['#D8B4FE','#6B21A8'])
    st.plotly_chart(fig, use_container_width=True)

elif menu == "تصدير التقارير":
    st.title("💾 تصدير التقارير")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 تحميل ملف CSV", csv, "customers_report.csv", "text/csv", use_container_width=True)

elif menu == "الاعدادات":
    st.title("⚙️ اعدادات النظام")
    company = st.text_input("اسم الشركة", "شركة الاتصالات")
    if st.button("💾 حفظ الاعدادات", use_container_width=True):
        st.success(f"✅ تم حفظ الاعدادات")

elif menu == "تسجيل الخروج":
    st.title("🚪 تسجيل الخروج")
    c1,c2 = st.columns(2)
    if c1.button("✅ نعم"): st.session_state.logged_in = False; st.rerun()
    if c2.button("❌ الغاء"): st.rerun()