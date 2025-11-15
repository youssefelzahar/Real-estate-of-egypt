import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import os

# Page configuration
st.set_page_config(
    page_title="نموذج تصنيف العقارات المصرية",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic support
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .segment-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid;
        color: white;
    }
    .segment-box h2,
    .segment-box h3,
    .segment-box p {
        color: white;
    }
    .segment-0 {
        background-color: #c62828;
        border-color: #b71c1c;
    }
    .segment-1 {
        background-color: #d32f2f;
        border-color: #c62828;
    }
    .segment-2 {
        background-color: #e53935;
        border-color: #d32f2f;
    }
    .segment-3 {
        background-color: #ef5350;
        border-color: #e53935;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .segment-box h3 {
        white-space: nowrap;
        word-break: keep-all;
    }
    .segment-box {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Segment descriptions in Arabic
# القيم مأخوذة من النموذج المدرب (cluster_summary)
SEGMENT_DESCRIPTIONS = {
    0: {
        "name": "عقارات عائلية ميسورة التكلفة",
        "name_en": "Affordable Family Homes",
        "description": "عقارات متوسطة الحجم مناسبة للعائلات، بأسعار معقولة وتوازن جيد بين المساحة والسعر",
        "avg_price": 8775326,  # 8,775,326 جنيه
        "avg_size_sqm": 297.63,
        "avg_bedrooms": 3.18,
        "avg_bathrooms": 2.79,
        "price_per_sqm": 29480  # محسوب من avg_price / avg_size_sqm
    },
    1: {
        "name": "أراضي وعقارات تجارية كبيرة",
        "name_en": "Large Land & Commercial Properties",
        "description": "أراضي وعقارات تجارية كبيرة جداً (متوسط 151,242 م²). السعر لكل متر مربع منخفض نسبياً بسبب الحجم الكبير جداً",
        "avg_price": 7972775,  # 7,972,775 جنيه
        "avg_size_sqm": 151242.5,
        "avg_bedrooms": 0.5,
        "avg_bathrooms": 2.0,
        "price_per_sqm": 53  # محسوب من avg_price / avg_size_sqm (52.72)
    },
    2: {
        "name": "قصور فاخرة متميزة",
        "name_en": "Premium Luxury Mansions",
        "description": "عقارات فاخرة عالية الجودة مع تشطيبات متميزة، غالباً في مواقع حصرية. أعلى سعر لكل متر مربع",
        "avg_price": 38965540,  # 38,965,540 جنيه
        "avg_size_sqm": 523.64,
        "avg_bedrooms": 4.60,
        "avg_bathrooms": 4.74,
        "price_per_sqm": 74320  # محسوب من avg_price / avg_size_sqm
    },
    3: {
        "name": "عقارات مدمجة بميزانية محدودة",
        "name_en": "Compact Budget Properties",
        "description": "وحدات أصغر حجماً بأسعار معتدلة، غالباً شقق أو تاون هاوس",
        "avg_price": 6613596,  # 6,613,596 جنيه
        "avg_size_sqm": 129.51,
        "avg_bedrooms": 1.94,
        "avg_bathrooms": 1.66,
        "price_per_sqm": 51050  # محسوب من avg_price / avg_size_sqm
    }
}

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load("segmentation_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"خطأ: لم يتم العثور على ملف النموذج: {e}")
        return None, None

@st.cache_data
def load_data():
    """Load the original dataset for finding similar properties"""
    try:
        df = pd.read_csv("egypt_real_estate_listings.csv")
        return df
    except FileNotFoundError:
        st.warning("ملف البيانات غير موجود. لن يتم عرض العقارات المشابهة.")
        return None

def preprocess_input(price, size_sqm, bedrooms, bathrooms):
    """Preprocess input features to match model requirements"""
    # Convert size_sqm to size_sqft (1 sqm = 10.764 sqft)
    size_sqft = size_sqm * 10.764
    
    # Calculate price per sqft
    price_per_sqft = price / size_sqft if size_sqft > 0 else 0
    
    # Create feature array matching the model's expected input
    features = np.array([[price, price_per_sqft, bedrooms, bathrooms, size_sqft, size_sqm]])
    
    return features

def predict_segment(model, scaler, features):
    """Predict the segment for given features"""
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict segment
    segment = model.predict(features_scaled)[0]
    
    # Ensure segment is a valid integer (0-3)
    segment = int(segment)
    if segment < 0 or segment > 3:
        segment = 0  # Default to segment 0 if invalid
    
    return segment

def find_similar_properties(df, price, size_sqm, bedrooms, bathrooms, segment, n=5):
    """Find similar properties from the dataset"""
    if df is None:
        return None
    
    try:
        df_filtered = df.copy()
        
        # Clean price column - remove commas and convert to numeric
        if 'price' in df_filtered.columns:
            df_filtered['price'] = df_filtered['price'].astype(str).str.replace(',', '', regex=False)
            df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce')
        
        # Clean bedrooms and bathrooms
        if 'bedrooms' in df_filtered.columns:
            df_filtered['bedrooms'] = df_filtered['bedrooms'].astype(str).str.replace('+', '', regex=False)
            df_filtered['bedrooms'] = pd.to_numeric(df_filtered['bedrooms'], errors='coerce')
        
        if 'bathrooms' in df_filtered.columns:
            df_filtered['bathrooms'] = df_filtered['bathrooms'].astype(str).str.replace('+', '', regex=False)
            df_filtered['bathrooms'] = pd.to_numeric(df_filtered['bathrooms'], errors='coerce')
        
        # Remove rows with missing critical data
        df_filtered = df_filtered.dropna(subset=['price', 'bedrooms', 'bathrooms'])
        
        # Filter by similar characteristics (within 30% price range)
        similar = df_filtered[
            (df_filtered['price'] >= price * 0.7) & (df_filtered['price'] <= price * 1.3) &
            (df_filtered['bedrooms'] == bedrooms) &
            (df_filtered['bathrooms'] == bathrooms)
        ]
        
        if len(similar) == 0:
            # Relax constraints - allow 1 bedroom/bathroom difference
            similar = df_filtered[
                (df_filtered['price'] >= price * 0.6) & (df_filtered['price'] <= price * 1.4) &
                (df_filtered['bedrooms'].between(max(0, bedrooms - 1), bedrooms + 1)) &
                (df_filtered['bathrooms'].between(max(0, bathrooms - 1), bathrooms + 1))
            ]
        
        if len(similar) > 0:
            # Sort by price difference - use .copy() to avoid SettingWithCopyWarning
            similar = similar.copy()
            similar['price_diff'] = abs(similar['price'] - price)
            similar = similar.sort_values('price_diff').head(n)
            
            # Select available columns
            cols_to_show = []
            for col in ['price', 'bedrooms', 'bathrooms', 'location', 'type']:
                if col in similar.columns:
                    cols_to_show.append(col)
            
            return similar[cols_to_show].to_dict('records')
        else:
            return None
    except Exception as e:
        st.warning(f"خطأ في البحث عن عقارات مشابهة: {e}")
        return None

def format_price(price):
    """Format price in Egyptian Pounds"""
    return f"{price:,.0f} جنيه"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 نموذج تصنيف العقارات المصرية</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 1.2em; color: #666;">
            آلة تصنيف ذكية لمساعدة السماسرة في تحديد فئة العقار المناسبة
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("⚠️ لا يمكن تحميل النموذج. يرجى التأكد من وجود ملفات segmentation_model.pkl و scaler.pkl")
        return
    
    # Load data for similar properties
    df = load_data()
    
    # Sidebar for input
    st.sidebar.header("📥 إدخال بيانات العقار")
    
    # Input fields
    location = st.sidebar.text_input("العنوان", value="التجمع الخامس", help="مثال: التجمع الخامس")
    price = st.sidebar.number_input(
        "السعر (جنيه مصري)", 
        min_value=0, 
        value=10000000, 
        step=100000,
        format="%d",
        help="أدخل سعر العقار بالجنيه المصري"
    )
    size_sqm = st.sidebar.number_input(
        "المساحة (متر مربع)", 
        min_value=0.0, 
        value=350.0, 
        step=10.0,
        help="أدخل مساحة العقار بالمتر المربع"
    )
    bedrooms = st.sidebar.number_input(
        "عدد غرف النوم", 
        min_value=0, 
        value=3, 
        step=1,
        help="عدد غرف النوم"
    )
    bathrooms = st.sidebar.number_input(
        "عدد الحمامات", 
        min_value=0, 
        value=3, 
        step=1,
        help="عدد الحمامات"
    )
    
    # Predict button
    if st.sidebar.button("🔍 تصنيف العقار", type="primary", use_container_width=True):
        # Preprocess input
        features = preprocess_input(price, size_sqm, bedrooms, bathrooms)
        
        # Predict segment
        segment = predict_segment(model, scaler, features)
        
        # Store results in session state
        st.session_state['segment'] = int(segment)  # Ensure it's an integer
        st.session_state['price'] = price
        st.session_state['size_sqm'] = size_sqm
        st.session_state['bedrooms'] = bedrooms
        st.session_state['bathrooms'] = bathrooms
        st.session_state['location'] = location
        st.session_state['features'] = features
        
        # Force rerun to update display
        st.rerun()
    
    # Display results
    if 'segment' in st.session_state:
        segment = st.session_state['segment']
        segment_info = SEGMENT_DESCRIPTIONS[segment]
        
        # Main results area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="segment-box segment-{segment}">
                <h2>📊 نتيجة التصنيف</h2>
                <h3>الفئة: {segment_info['name']} <span style="white-space: nowrap;">({segment_info['name_en']})</span></h3>
                <p><strong>الوصف:</strong> {segment_info['description']}</p>
                <p><strong>رقم الفئة:</strong> {segment}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("السعر المدخل", format_price(st.session_state['price']))
            st.metric("المساحة", f"{st.session_state['size_sqm']:.0f} م²")
            st.metric("غرف النوم", f"{st.session_state['bedrooms']}")
            st.metric("الحمامات", f"{st.session_state['bathrooms']}")
        
        # Expected price and segment statistics
        st.markdown("---")
        st.subheader("💰 السعر المتوقع ومعلومات الفئة")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "متوسط سعر الفئة",
                format_price(segment_info['avg_price']),
                help="متوسط سعر العقارات في هذه الفئة"
            )
        
        with col2:
            st.metric(
                "متوسط المساحة",
                f"{segment_info['avg_size_sqm']:.0f} م²",
                help="متوسط مساحة العقارات في هذه الفئة"
            )
        
        with col3:
            st.metric(
                "متوسط غرف النوم",
                f"{segment_info['avg_bedrooms']:.1f}",
                help="متوسط عدد غرف النوم في هذه الفئة"
            )
        
        with col4:
            st.metric(
                "السعر لكل م²",
                format_price(segment_info['price_per_sqm']),
                help="متوسط السعر لكل متر مربع في هذه الفئة"
            )
        
        # Price comparison
        st.markdown("---")
        st.subheader("📈 مقارنة السعر")
        
        input_price = st.session_state['price']
        avg_price = segment_info['avg_price']
        price_diff = ((input_price - avg_price) / avg_price) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            if abs(price_diff) < 10:
                st.success(f"✅ السعر مناسب للفئة (الفرق: {price_diff:+.1f}%)")
            elif price_diff > 10:
                st.warning(f"⚠️ السعر أعلى من متوسط الفئة ({price_diff:+.1f}%)")
            else:
                st.info(f"ℹ️ السعر أقل من متوسط الفئة ({price_diff:+.1f}%)")
        
        with col2:
            st.metric(
                "الفرق عن متوسط الفئة",
                f"{price_diff:+.1f}%",
                delta=f"{price_diff:+.1f}%"
            )
        
        # Similar properties
        if df is not None:
            st.markdown("---")
            st.subheader("🏘️ عقارات مشابهة")
            
            similar = find_similar_properties(
                df, 
                st.session_state['price'], 
                st.session_state['size_sqm'],
                st.session_state['bedrooms'],
                st.session_state['bathrooms'],
                segment
            )
            
            if similar:
                for i, prop in enumerate(similar, 1):
                    with st.expander(f"عقار مشابه #{i}: {format_price(prop.get('price', 0))}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**السعر:** {format_price(prop.get('price', 0))}")
                            st.write(f"**الموقع:** {prop.get('location', 'غير محدد')}")
                        with col2:
                            st.write(f"**النوع:** {prop.get('type', 'غير محدد')}")
                            st.write(f"**غرف النوم:** {prop.get('bedrooms', 'N/A')}")
                            st.write(f"**الحمامات:** {prop.get('bathrooms', 'N/A')}")
            else:
                st.info("لم يتم العثور على عقارات مشابهة في قاعدة البيانات.")
        
        # Feature importance visualization
        st.markdown("---")
        st.subheader("📊 تحليل الخصائص")
        
        # Calculate price per sqm for input
        input_price_per_sqm = st.session_state['price'] / st.session_state['size_sqm'] if st.session_state['size_sqm'] > 0 else 0
        
        comparison_data = {
            'المساحة (م²)': [st.session_state['size_sqm'], segment_info['avg_size_sqm']],
            'غرف النوم': [st.session_state['bedrooms'], segment_info['avg_bedrooms']],
            'الحمامات': [st.session_state['bathrooms'], segment_info['avg_bathrooms']],
        }
        
        comparison_df = pd.DataFrame(comparison_data, index=['العقار المدخل', 'متوسط الفئة'])
        st.bar_chart(comparison_df.T)
    
    else:
        # Initial state - show instructions
        st.info("""
        👈 **ابدأ من هنا!**
        
        استخدم القائمة الجانبية لإدخال بيانات العقار:
        - العنوان
        - السعر (بالجنيه المصري)
        - المساحة (بالمتر المربع)
        - عدد غرف النوم
        - عدد الحمامات
        
        ثم اضغط على زر "تصنيف العقار" للحصول على النتائج.
        """)
        
        # Show segment descriptions
        st.markdown("---")
        st.subheader("📋 فئات العقارات")
        
        for seg_num, seg_info in SEGMENT_DESCRIPTIONS.items():
            with st.expander(f"الفئة {seg_num}: {seg_info['name']} ({seg_info['name_en']})"):
                st.write(f"**الوصف:** {seg_info['description']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("متوسط السعر", format_price(seg_info['avg_price']))
                with col2:
                    st.metric("متوسط المساحة", f"{seg_info['avg_size_sqm']:.0f} م²")
                with col3:
                    st.metric("السعر/م²", format_price(seg_info['price_per_sqm']))

if __name__ == "__main__":
    main()

