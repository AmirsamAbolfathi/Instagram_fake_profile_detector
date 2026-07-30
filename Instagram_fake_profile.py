import pickle as pk
import base64
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Instagram Fake Profile Detector",
    page_icon="favicon.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)


def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg_base64 = get_base64("bg.png")

st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(10, 10, 15, 0.75), rgba(10, 10, 15, 0.75)),
            url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1 {{
        color: #E1306C;
        text-align: center;
        font-weight: 800;
    }}
    .subtitle {{
        text-align: center;
        color: #e6e6e6;
        margin-bottom: 30px;
    }}

    div[data-testid="stForm"] {{
        background-color: rgba(22, 27, 34, 0.85);
        padding: 25px 25px 10px 25px;
        border-radius: 16px;
        border: 1px solid #2a2f3a;
        backdrop-filter: blur(4px);
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #f58529, #dd2a7b, #8134af, #515bd4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
        transition: 0.2s;
    }}
    .stButton>button:hover {{
        opacity: 0.85;
        transform: scale(1.01);
    }}

    div[data-testid="stMetric"] {{
        background-color: rgba(22, 27, 34, 0.85);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #2a2f3a;
    }}

    h4, .stMarkdown p {{
        color: #f0f0f0;
    }}
    </style>
""", unsafe_allow_html=True)


with open("scalar.pkl", "rb") as file:
    scalar = pk.load(file)

with open("model_forest.pkl", "rb") as file:
    model = pk.load(file)


def nums_length_uname(uname):
    list_nums_username = [char for char in uname if char.isdigit()]
    len_nums_username = len(list_nums_username)
    len_username = len(uname)
    return len_nums_username / len_username if len_username else 0


def fname_words(fname):
    return len(fname.split())


def nums_length_fname(fname):
    list_nums_fullname = [char for char in fname if char.isdigit()]
    len_fullname = len(fname)
    len_nums_fullname = len(list_nums_fullname)
    return len_nums_fullname / len_fullname if len_fullname else 0


def fname_uname(uname, fname):
    return 1 if uname == fname else 0


def descrp_length(description):
    return len(description)


# ---------- Header ----------
st.title("🕵️‍♂️ Instagram Fake Profile Detector")
st.markdown('<p class="subtitle">Enter the account details below and let the model decide</p>', unsafe_allow_html=True)

profile_pic_dict = {"Yes": 1, "No": 0}
external_URL_dict = {"Yes": 1, "No": 0}
private_dict = {"Yes": 1, "No": 0}
fake_dict = {1: "Yes", 0: "No"}

# ---------- Form ----------
with st.form("profile_form"):
    st.markdown("#### 👤 Basic info")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full name", value="")
    with col2:
        username = st.text_input("Username", value="")

    description = st.text_area("Account bio / description", value="")

    st.markdown("#### 📊 Account stats")
    col3, col4, col5 = st.columns(3)
    with col3:
        posts = st.number_input("Posts", min_value=0, step=1)
    with col4:
        followers = st.number_input("Followers", min_value=0, step=1)
    with col5:
        follows = st.number_input("Following", min_value=0, step=1)

    st.markdown("#### ⚙️ Account settings")
    col6, col7, col8 = st.columns(3)
    with col6:
        profile_pic_key = st.radio("Profile picture?", list(profile_pic_dict.keys()))
    with col7:
        private_key = st.radio("Private account?", list(private_dict.keys()))
    with col8:
        external_URL_key = st.radio("Has external URL?", list(external_URL_dict.keys()))

    submitted = st.form_submit_button("🔍 Analyze account")

# ---------- Prediction ----------
if submitted:
    with st.spinner("Analyzing profile..."):
        profile_pic = profile_pic_dict[profile_pic_key]
        numslength_username = nums_length_uname(username)
        fullname_words = fname_words(full_name)
        nums_length_fullname = nums_length_fname(full_name)
        name_username = fname_uname(full_name, username)
        description_length = descrp_length(description)
        external_URL = external_URL_dict[external_URL_key]
        private = private_dict[private_key]

        data = np.array([[profile_pic, numslength_username, fullname_words,
                           nums_length_fullname, name_username, description_length,
                           external_URL, private, posts, followers, follows]])
        data = scalar.transform(data)
        result = model.predict(data)
        prediction = result.item()

    st.markdown("---")
    st.markdown("#### 📈 Account overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Posts", int(posts))
    m2.metric("Followers", int(followers))
    m3.metric("Following", int(follows))

    st.markdown("#### 🧾 Result")
    if prediction == 0:
        st.success("✅ This profile looks **REAL**.")
    else:
        st.error("⚠️ This profile looks **FAKE**.")

    # If the model supports probabilities, show a confidence score
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data)[0]
            confidence = max(proba) * 100
            st.progress(int(confidence))
            st.caption(f"Model confidence: {confidence:.1f}%")
        except Exception:
            pass
