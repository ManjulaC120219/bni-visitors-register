import streamlit as st
import pandas as pd
from supabase import create_client, Client
from datetime import datetime, timedelta, date
import time
import logging
from typing import Optional, Dict, Any, Tuple
from BNI_Visitors_data_extraction import extract_data_from_image
import traceback

from datetime import date
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="BNI Brilliance Visitors Register",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Supabase configuration
SUPABASE_URL = "https://dvzpeyupbyqkehksvmpc.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR2enBleXVwYnlxa2Voa3N2bXBjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3NzQwNzMsImV4cCI6MjA3MTM1MDA3M30.cKw87wSBjpqBMp42cFh5oOqRLfwOpzYysEasJ2T8llc"

# Error handling utilities
class AppError(Exception):
    """Base exception for application errors"""
    pass

class DatabaseError(AppError):
    """Database related errors"""
    pass

class ValidationError(AppError):
    """Input validation errors"""
    pass

class AuthenticationError(AppError):
    """Authentication related errors"""
    pass
# Enhanced error tracking
def increment_error_count():
    """Track errors for monitoring"""
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    st.session_state.error_count += 1
    st.session_state.last_error = datetime.now()

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0


def show_error_details(error_message):
    """Display error details in a user-friendly way"""
    with st.expander("🔍 Error Details"):
        st.code(error_message)

def upload_and_process_image(uploaded_file):
    """Upload image and process it to extract table data."""
    # Save uploaded file to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract data using the function from data_extraction.py
    extracted_data = extract_data_from_image("temp_image.png")

    return extracted_data

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            st.error(f"Authentication failed: {str(e)}")
            logger.error(f"Authentication error in {func.__name__}: {e}")
            return None
        except ValidationError as e:
            st.error(f"Invalid input: {str(e)}")
            logger.error(f"Validation error in {func.__name__}: {e}")
            return None
        except DatabaseError as e:
            st.error(f"Database error: {str(e)}")
            logger.error(f"Database error in {func.__name__}: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper


def validate_input(field_name: str, value: Any, required: bool = False, min_length: int = 0) -> bool:
    """Validate input fields"""
    if required and (value is None or str(value).strip() == ""):
        raise ValidationError(f"{field_name} is required")

    if isinstance(value, str) and len(value.strip()) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")

    return True

# Initialize Supabase client with error handling
@st.cache_resource
def init_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        client.table('bni_visitors_details').select('id').limit(1).execute()
        return client
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

try:
    supabase: Client = init_supabase()
except DatabaseError:
    st.error("❌ Database Connection Failed")
    st.info("Please check your Supabase credentials and network connection.")
    st.code("""
    Troubleshooting steps:
    1. Verify Supabase URL and API key
    2. Check network connectivity
    3. Ensure database tables exist
    4. Check Supabase service status
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .delete-confirmation {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .success-message {
        background-color: #efe;
        border: 1px solid #cfc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .error-container {
        background-color: #fee;
        border-left: 4px solid #f56565;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def save_data_to_supabase(dataframe: pd.DataFrame):
    """Save the edited dataframe back to Supabase."""
    success_count = 0
    error_count = 0
    errors = []

    try:
        for _, row in dataframe.iterrows():
            # Clean and convert data types
            def safe_convert_to_int(value):
                """Convert value to int, return None if empty or invalid"""
                if pd.isna(value) or value == "" or value is None:
                    return None
                try:
                    return int(str(value))  # Handle float strings like "123.0"
                except (ValueError, TypeError):
                    return None

            def safe_convert_to_string(value):
                """Convert value to string, return empty string if None/NaN"""
                if pd.isna(value) or value is None:
                    return None
                return str(value)

            from datetime import datetime
            import pandas as pd

            def safe_convert_to_date(value, date_format=None):
                """
                Convert a value to a datetime object.

                - If `date_format` is provided, it will use that format to parse the date.
                - If the value is empty, None, or not a valid date, return None.
                - Handles strings, pandas Timestamps, and datetime objects.
                """
                if pd.isna(value) or value in ("", None):
                    return None

                # If it's already a datetime object or pandas Timestamp, return as datetime
                if isinstance(value, (datetime, pd.Timestamp)):
                    return value.to_pydatetime()

                try:
                    if date_format:
                        return datetime.strptime(str(value), date_format)
                    else:
                        return pd.to_datetime(value, errors='coerce').to_pydatetime()
                except (ValueError, TypeError):
                    return None

            member_data = {
                "name": safe_convert_to_string(row["Name"]),
                "company_name": safe_convert_to_string(row["company name"]),
                "category": safe_convert_to_string(row["category"]),
                "invited_by": safe_convert_to_string(row["invited by"]),
                "fees": safe_convert_to_int(row["fees"]),
                "payment_mode": safe_convert_to_string(row["payment mode"]),
                "visit_date":safe_convert_to_date(row["visit_date"]),
                "created_at": datetime.now().isoformat(),
            }

            # Check if record already exists
            existing = supabase.table("bni_visitors_details").select("*").eq("name", member_data["name"]).execute()

            if existing.data:
                # Update existing record
                response = supabase.table("bni_visitors_details").update(member_data).eq("name",
                                                                                          member_data["name"]).execute()
            else:
                # Insert new record
                response = supabase.table("bni_visitors_details").insert(member_data).execute()

            # Check if the response contains data (success)
            if response.data:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"No data returned for {row['Name']}")

        # Show summary message after all operations
        if success_count > 0:
            st.success(f"Successfully saved {success_count} member(s) to database!")

        if error_count > 0:
            st.error(f"Failed to save {error_count} member(s). Errors: {'; '.join(errors)}")

    except Exception as e:
        st.error(f"Database operation failed: {str(e)}")

def debug_supabase_connection():
    """Debug function to test Supabase connection"""
    try:
        supabase = init_supabase()
        if not supabase:
            return "❌ Supabase client is None"

        # Test basic connection
        response = supabase.table("bni_visitors_details").select("*").limit(1).execute()
        return f"✅ Connection OK. Sample data: {response.data}"

    except Exception as e:
        return f"❌ Connection failed: {str(e)}"

# Alternative simplified save function for testing
def save_data_simple_test(dataframe: pd.DataFrame):
    """Simplified version for testing with correct schema"""
    try:
        supabase = init_supabase()

        # Helper functions
        def safe_int_convert(value):
            if pd.isna(value) or value == "" or value is None:
                return None
            try:
                return int(str(value))
            except (ValueError, TypeError):
                return None

        def safe_date_convert(value, date_format=None):
            """
            Convert a value to a datetime object.

            - If `date_format` is provided, it will use that format to parse the date.
            - If the value is empty, None, or not a valid date, return None.
            - Handles strings, pandas Timestamps, and datetime objects.
            """
            if pd.isna(value) or value in ("", None):
                return None

            # If it's already a datetime object or pandas Timestamp, return as datetime
            if isinstance(value, (datetime, pd.Timestamp)):
                return value.to_pydatetime()

            try:
                if date_format:
                    return datetime.strptime(str(value), date_format)
                else:
                    return pd.to_datetime(value, errors='coerce').to_pydatetime()
            except (ValueError, TypeError):
                return None

        # Test with just one record
        first_row = dataframe.iloc[0]
        test_data = {
            "name": str(first_row.get("Name", "")).strip(),
            "company_name": str(first_row.get("Company Name", "")).strip(),
            "category": str(first_row.get("Category", "")).strip(),
            "invited_by": str(first_row.get("Invited by", "")).strip(),
            "fees": safe_int_convert(first_row.get("Fees", "")),  # int8
            "payment_mode": str(first_row.get("Payment Mode", "")).strip(),
            "visit_date":safe_date_convert(first_row.get("Date", "")),
            "created_at": datetime.now().isoformat()
        }

        st.write("Testing with data:", test_data)
        st.write("Payment value type:", type(test_data["fees"]))

        response = supabase.table("bni_visitors_details").insert(test_data).execute()

        st.write("Response:", response.data)
        st.write("Error:", getattr(response, 'error', 'None'))

        return response.data is not None

    except Exception as e:
        st.error(f"Test failed: {str(e)}")
        st.write(traceback.format_exc())
        return False
# Database helper functions with enhanced error handling
#def hash_password(password: str) -> str:
    #"""Hash password using SHA-256 with input validation"""
    #try:
        #validate_input("Password", password, required=True, min_length=1)
        #return hashlib.sha256(password.encode()).hexdigest()
    #except Exception as e:
        #logger.error(f"Password hashing failed: {e}")
        #raise ValidationError("Failed to process password")


#@handle_error
#def authenticate_user(username: str, password: str) -> bool:
    #"""Authenticate admin user with comprehensive error handling"""
    #try:
        #validate_input("Username", username, required=True)
        #validate_input("Password", password, required=True)

        #password_hash = hash_password(password)

        # Enhanced query with better error handling
        #response = supabase.table('admins').select('*').eq('username', username.strip()).execute()

        #if not response.data:
            #logger.warning(f"No admin found with username: {username}")
            #raise AuthenticationError("Invalid credentials")

        #stored_user = response.data[0]
        #stored_hash = stored_user.get('password_hash', '')

        #if not stored_hash:
            #logger.error("No password hash found in database")
            #raise DatabaseError("User account corrupted - no password hash")

        #if stored_hash == password_hash:
            #logger.info(f"Successful login for user: {username}")
            #return True
        #else:
            #logger.warning(f"Password mismatch for user: {username}")
            #raise AuthenticationError("Invalid credentials")

    #except (ValidationError, AuthenticationError, DatabaseError):
        #raise
    #except Exception as e:
        #logger.error(f"Unexpected authentication error: {e}")
        #raise AuthenticationError("Authentication service temporarily unavailable")


#@handle_error
#def ensure_admin_exists() -> bool:
    #"""Create default admin if not exists with proper error handling"""
    #try:
        # Check if admin table exists and is accessible
        #response = supabase.table('admins').select('username').limit(1).execute()

        # Check for existing admin
        #admin_response = supabase.table('admins').select('*').eq('username', 'admin').execute()

        #if len(admin_response.data) == 0:
            # Create new admin with the correct password hash
            #default_password = 'admin123'
            #admin_data = {
                #'username': 'admin',
                #'password_hash': hash_password(default_password),
                #'created_at': datetime.now().isoformat()
            #}

            #result = supabase.table('admins').insert(admin_data).execute()

            #if result.data:
                #logger.info("Default admin created successfully")
                #st.success("✅ Default admin created successfully!")
                #st.info(f"Username: admin | Password: {default_password}")
                #return True
            #else:
                #raise DatabaseError("Failed to create admin account")

        #logger.info("Admin account already exists")
        #return False

    #except Exception as e:
        #logger.error(f"Admin creation failed: {e}")
        #raise DatabaseError(f"Could not create/verify admin account: {str(e)}")



@handle_error
def get_all_members() -> pd.DataFrame:
    """Fetch all members from database with error handling"""
    try:
        response = supabase.table('bni_visitors_details').select('*').order('created_at', desc=True).execute()

        if response.data is None:
            logger.warning("No data returned from members query")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        logger.info(f"Successfully fetched {len(df)} members")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch members: {e}")
        raise DatabaseError(f"Could not retrieve members: {str(e)}")

@handle_error
def add_member(name: str,  company_name: str, category: str, invited_by: str, fees: int, payment_mode: str, visit_date: datetime) -> bool:
    """Add new member with comprehensive validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if fees <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Prepare member data
        member_data = {
            'name': name.strip(),
            'company_name': company_name.strip() if company_name else '',
            'category': category.strip(),
            'invited_by': invited_by.strip(),
            'fees': fees,
            'payment_mode':payment_mode.strip(),
            'visit_date': visit_date.isoformat(),
            'created_at': datetime.now().isoformat()
        }

        # Check for duplicate names (optional warning)
        existing_response = supabase.table('bni_visitors_details').select('name').eq('name', name.strip()).execute()
        if existing_response.data:
            st.warning(f"⚠️ A member named '{name}' already exists. Adding anyway...")

        # Insert member
        response = supabase.table('bni_visitors_details').insert(member_data).execute()

        if response.data:
            logger.info(f"Successfully added visitor: {name}")
            return True
        else:
            raise DatabaseError("Failed to insert visitor data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error adding visitor: {e}")
        raise DatabaseError(f"Could not add visitor: {str(e)}")


@handle_error
def update_member(member_id: int, name : str, company_name: str, category: str, invited_by: str, fees: int, payment_mode: str, visit_date: datetime) -> bool:
    """Update member with validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if fees <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Check if member exists
        check_response = supabase.table('bni_visitors_details').select('id').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        # Prepare update data
        member_data = {
            'name': name.strip(),
            'company_name': company_name.strip() if company_name else '',
            'category': category.strip(),
            'invited_by': invited_by.strip(),
            'fees': fees,
            'payment_mode': payment_mode.strip(),
            'visit_date':visit_date
        }
        # Update member
        response = supabase.table('bni_visitors_details').update(member_data).eq('id', member_id).execute()

        if response.data:
            logger.info(f"Successfully updated member ID: {member_id}")
            return True
        else:
            raise DatabaseError("Failed to update member data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating member: {e}")
        raise DatabaseError(f"Could not update member: {str(e)}")

@handle_error
def delete_member(member_id: int) -> bool:
    """Delete member with error handling"""
    try:
        # Check if member exists
        check_response = supabase.table('bni_visitors_details').select('id, name').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        member_name = check_response.data[0]['name']

        # Delete member
        response = supabase.table('bni_visitors_details').delete().eq('id', member_id).execute()

        if response.data is not None:  # Supabase delete returns data on success
            logger.info(f"Successfully deleted member: {member_name} (ID: {member_id})")
            return True
        else:
            raise DatabaseError("Failed to delete member")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting member: {e}")
        raise DatabaseError(f"Could not delete member: {str(e)}")

def get_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate member statistics with error handling"""
    try:
        if df.empty:
            return {'total_members': 0, 'total_payment': 0}

        # Ensure payment column exists and is numeric
        if 'fees' not in df.columns:
            logger.warning("Fees column missing from dataframe")
            return {'total_members': len(df), 'total_payment': 0}

        # Handle non-numeric payment values
        numeric_payments = pd.to_numeric(df['fees'], errors='coerce').fillna(0)

        return {
            'total_members': len(df),
            'total_payment': numeric_payments.sum(),
        }
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return {'total_members': 0, 'total_payment': 0}

@handle_error
#def reset_admin_password() -> bool:
    #"""Reset admin password to default with proper hashing"""
    #try:
        #default_password = 'admin123'
        #new_hash = hash_password(default_password)

        # First, delete any existing admin to avoid conflicts
        #supabase.table('admins').delete().eq('username', 'admin').execute()

        # Create fresh admin with correct hash
        #admin_data = {
            #'username': 'admin',
            #'password_hash': new_hash,
            #'created_at': datetime.now().isoformat()
        ##}

        #create_response = supabase.table('admins').insert(admin_data).execute()

        #if create_response.data:
            #logger.info("Admin password reset successfully")
            #st.success("✅ Admin account reset successfully!")
            #st.info("Username: admin | Password: admin123")
            #return True
        #else:
            #raise DatabaseError("Failed to create admin account")

    #except Exception as e:
        #logger.error(f"Password reset failed: {e}")
        #raise DatabaseError(f"Could not reset password: {str(e)}")

def init_session_state():
    """Initialize session state variables with error recovery"""
    try:
        default_states = {
            'logged_in': False,
            'show_add_form': False,
            'edit_member': None,
            'delete_confirmation': None,
            'search_query': "",
            'error_count': 0,
            'last_error': None,
            'selected_dates_calendar': set(),
            'amount': 0.0,
            'selected_thursdays': []
        }
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        # Force reset session state on critical failure
        for key in list(st.session_state.keys()):
            del st.session_state[key]


# Session state management with error recovery
def show_error_details(error_message: str):
    """Show detailed error information for debugging"""
    with st.expander("🔧 Error Details (Click to expand)"):
        st.error(error_message)
        st.code(f"""
        Error Count: {st.session_state.get('error_count', 0)}
        Last Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Troubleshooting Tips:
        1. Refresh the page and try again
        2. Check your internet connection
        3. Verify database tables exist in Supabase
        4. Clear browser cache if issues persist
        """)

@handle_error
def search_members(query: str) -> pd.DataFrame:
    """Search members with error handling"""
    try:
        validate_input("Search query", query, required=True, min_length=1)

        # Sanitize search query
        clean_query = query.strip().replace("'", "''")  # Basic SQL injection prevention

        response = supabase.table('bni_visitors_details').select('*').or_(

            f'name.ilike.%{clean_query}%,toa.ilike.%{clean_query}%'
        ).order('created_at', desc=True).execute()

        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        logger.info(f"Search for '{query}' returned {len(df)} results")
        return df

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise DatabaseError(f"Search failed: {str(e)}")

def process_supabase_data_simple(raw_data):
    """Process raw Supabase data and calculate total fee per individual (no Thursday logic)"""
    try:
        from datetime import datetime

        st.write(f"🔍 **Processing {len(raw_data)} records from Supabase...**")

        payment_by_person = {}

        for record in raw_data:
            name = record.get('name', 'Unknown').strip().title()
            fees = record.get('fees', 0)


            # Accumulate fee per person
            if name in payment_by_person:
                payment_by_person[name] += fees
            else:
                payment_by_person[name] = fees

        # Format as list of dicts
        processed_data = [
            {"name": name, "total_payment": total}
            for name, total in payment_by_person.items()
        ]

        st.write(f"📊 **Successfully processed {len(processed_data)} individuals**")
        return processed_data

    except Exception as e:
        st.error(f"❌ Error processing Supabase data: {str(e)}")
        return []

def display_payment_summary(selected_thursday, weekly_amount):
    """Display payment summary for a single Thursday"""
    try:
        # ✅ Check if data is loaded
        if 'individuals_data' not in st.session_state or st.session_state.individuals_data is None:
            st.info("No payment data loaded.")
            return

        individuals = st.session_state.individuals_data

        # ✅ If DataFrame, convert to list of dicts
        if isinstance(individuals, pd.DataFrame):
            individuals = individuals.to_dict(orient="records")

        st.write(f"### 📅 Payment Summary for: **{selected_thursday.strftime('%B %d, %Y')}**")
        #st.write(f"🧾 Weekly Payment Amount: Rs. {weekly_amount:,.2f}")

        summary_data = []
        total_collected = 0
        attendees_count = 0


        for individual in individuals:
            fees = individual.get("fees", 0)
            if isinstance(fees, list):
                total_paid = sum(fees)
            elif isinstance(fees, (int, float)):
                total_paid = fees
            else:
                try:
                    total_paid = float(fees)
                except:
                    total_paid = 0

            if total_paid > 0:
                attendees_count += 1
                total_collected += total_paid

            summary_data.append({
                "Name": individual.get("name", "Unknown"),
                "Paid": f"Rs. {total_paid:,.2f}"
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🧑‍🤝‍🧑 Total Attendees", attendees_count)
        with col2:
            st.metric("💰 Total Collected", f"Rs. {total_collected:,.2f}")

    except Exception as e:
        st.error(f"❌ Error displaying payment summary: {str(e)}")

# Solution 2: Force sidebar visibility in show_sidebar function
def show_sidebar():
    """Show sidebar with file upload and other options"""
    try:
        # Force sidebar to be visible
        st.sidebar.markdown("")  # This ensures sidebar is created

        with st.sidebar:
            # Error monitoring
            if st.session_state.get('error_count', 0) > 0:
                st.warning(f"⚠️ {st.session_state.error_count} error(s) occurred this session")

            # Image Processing Section
            st.header("📂 Image Processing")

            # Option selector for upload or capture
            capture_option = st.radio(
                "Choose input method:",
                options=["Upload File", "Capture Image"],
                key="input_method"
            )

            uploaded_file = None
            captured_image = None

            if capture_option == "Upload File":
                # File upload option
                uploaded_file = st.file_uploader(
                    "Upload an image file",
                    type=["jpg", "jpeg", "png"],
                    help="Select an image file from your device"
                )

                if uploaded_file is not None:
                    st.success("✅ File uploaded!")
                    # Store in session state for dashboard processing
                    st.session_state.image_to_process = uploaded_file
                    st.session_state.image_source = "upload"

            else:  # Capture Image
                # Camera capture option
                st.info("📷 Use camera to capture")
                captured_image = st.camera_input(
                    "Take a picture",
                    help="Click to take a picture using your device's camera"
                )

                if captured_image is not None:
                    st.success("✅ Image captured!")
                    # Store in session state for dashboard processing
                    st.session_state.image_to_process = captured_image
                    st.session_state.image_source = "capture"

            # Process button
            if st.session_state.get('image_to_process') is not None:
                if st.button("🔄 Process Image", use_container_width=True, type="primary"):
                    with st.spinner("Processing..."):
                        try:
                            # Process the image and extract the data
                            extracted_data = upload_and_process_image(st.session_state.image_to_process)

                            # Store extracted data in session state
                            if isinstance(extracted_data, list):
                                extracted_data = pd.DataFrame(extracted_data)

                            st.session_state.extracted_data = extracted_data
                            st.session_state.data_processed = True
                            st.success("✅ Data extracted!")
                            st.rerun()  # Refresh to show data in main area

                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            st.session_state.extracted_data = None
                            st.session_state.data_processed = False

            # Clear processed data button
            if st.session_state.get('data_processed'):
                if st.button("🗑️ Clear Data", use_container_width=True):
                    st.session_state.image_to_process = None
                    st.session_state.extracted_data = None
                    st.session_state.data_processed = False
                    st.rerun()

            st.divider()

            # Action buttons with error handling
            if st.button("➕ Add New Visitors", use_container_width=True):
                st.session_state.show_add_form = True
                st.session_state.edit_member = None
                st.session_state.show_calendar = False  # <-- Reset calendar state
                st.rerun()

            if st.button("📅 Calendar", use_container_width=True):
                st.session_state.show_calendar = True
                st.session_state.show_add_form = False
                st.session_state.edit_member = None
                st.rerun()

            #if st.button("🚪 Logout", use_container_width=True):
                #for key in list(st.session_state.keys()):
                    #del st.session_state[key]
                #st.rerun()

    except Exception as e:
        st.sidebar.error(f"Sidebar error: {str(e)}")
        logger.error(f"Sidebar error: {e}")

# Solution 4: Quick Debug - Add this to your current main execution
def debug_sidebar():
    """Quick debug function to force sidebar visibility"""
    with st.sidebar:
        st.write("🔧 Debug: Sidebar is working!")
        st.write("If you see this, sidebar is functional")

# Solution 5: Modified show_dashboard to ensure sidebar is called

#def #verify_password_hash():
    #"""Test function to verify password hashing"""
    #test_password = "admin123"
    #expected_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
    #actual_hash = hash_password(test_password)
    #print(f"Expected: {expected_hash}")
    #print(f"Actual:   {actual_hash}")
    #print(f"Match:    {expected_hash == actual_hash}")
    #return expected_hash == actual_hash

# Add this function temporarily for debugging

# Enhanced login page with error recovery
#def show_login():
    #st.markdown("""
    #<div style="text-align: center; padding: 2rem;">
        #<h1>🛡️ BNI Brilliance Admin Login</h1>
        #<p>Enter your credentials to access the member management system</p>
    #</div>
    #""", unsafe_allow_html=True)

    #col1, col2, col3 = st.columns([1, 2, 1])

    #with col2:
        # Show system status
        #try:
            # Quick health check
            #health_check = supabase.table('image_data_extraction').select('id').limit(1).execute()
            #st.success("🟢 Database connection healthy")
        #except Exception as e:
            #st.error("🔴 Database connection issues")
            #show_error_details(str(e))

        #with st.form("login_form"):
            #username = st.text_input("Username", placeholder="Enter admin username")
            #password = st.text_input("Password", type="password", placeholder="Enter password")
            #submitted = st.form_submit_button("Login", use_container_width=True)

            #if submitted:
                #with st.spinner("Authenticating..."):
                    #if authenticate_user(username, password):
                        #st.session_state.logged_in = True
                        #st.session_state.error_count = 0  # Reset error count on success
                        #st.success("✅ Login successful!")
                        #time.sleep(1)
                        #st.rerun()

        #st.info("""
        #**Default Credentials:**
        #Username: `admin`
        #Password: `admin123`
        #""")

        # Emergency reset button
        #st.divider()
        #st.subheader("🆘 Having trouble logging in?")

        #col_reset1, col_reset2 = st.columns(2)

        #with col_reset1:
            #if st.button("🔄 Reset Admin Password", help="Resets admin password to default"):
                #if reset_admin_password():
                    #st.success("✅ Admin password reset to 'admin123'")
                    #time.sleep(2)
                    #st.rerun()

        #with col_reset2:
            #if st.button("🔧 Create Admin Account", help="Creates admin account if missing"):
                #if ensure_admin_exists():
                    #time.sleep(1)
                    #st.rerun()

# Enhanced dashboard with error monitoring
def show_dashboard():
    """Main dashboard function - only shows dashboard content, no sidebar"""
    st.title("🏢 BNI Brilliance Visitors Register")

    # Initialize session state variables
    if "show_calendar" not in st.session_state:
        st.session_state.show_calendar = False

    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False

    if "edit_member" not in st.session_state:
        st.session_state.edit_member = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    if "selected_dates_calendar" not in st.session_state:
        st.session_state.selected_dates_calendar = set()

    if "delete_confirmation" not in st.session_state:
        st.session_state.delete_confirmation = None

    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None

    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False

    # Display extracted data section (only if data exists)
    if st.session_state.get('data_processed') and st.session_state.get('extracted_data') is not None:

        st.header("📝 Extracted Data - Review & Edit")

        # Show source information
        source_info = st.session_state.get('image_source', 'unknown')
        if source_info == "upload":
            st.info("📁 Data extracted from uploaded file")
        else:
            st.info("📷 Data extracted from captured image")

        extracted_data = st.session_state.extracted_data

        # Ensure extracted_data is a pandas DataFrame
        if isinstance(extracted_data, list):
            extracted_data = pd.DataFrame(extracted_data)

        # Display extracted data as a dataframe and allow editing
        edited_data = extracted_data.copy()

        for idx, row in edited_data.iterrows():
            st.subheader(f"Editing Row {idx + 1}")
            edited_data.at[idx, "Name"] = st.text_input(f"Name", value=row["Name"], key=f"name_{idx}")
            edited_data.at[idx, "Company Name"] = st.text_input(f"Company Name", value=row["Company Name"], key=f"company_name_{idx}")
            edited_data.at[idx, "Category"] = st.text_input(f"Category", value=row["Category"], key=f"category_{idx}")
            edited_data.at[idx, "Invited by"] = st.text_input(f"Invited by", value=row["Invited by"],
                                                              key=f"invited_by_{idx}")
            edited_data.at[idx, "Fees"] = st.text_input(f"Fees", value=str(row["Fees"]), key=f"fees_{idx}")
            edited_data.at[idx, "Payment Mode"] = st.text_input(f"Payment Mode", value=row["Payment Mode"],
                                                                key=f"payment_mode_{idx}")
            edited_data.at[idx, "Date"] = st.text_input(f"Date", value=row["Date"],
                                                                key=f"visit_date_{idx}")

        # Display the editable table as a dataframe (not editable directly)
        st.dataframe(edited_data)

        # Save the edited data back to Supabase
        if st.button("Save to Database", key="save_to_db_btn"):
            save_data_to_supabase(edited_data)

        # Handle delete confirmation with error handling
    if st.session_state.delete_confirmation:
        member_to_delete = st.session_state.delete_confirmation

        st.markdown(f"""
                <div class="delete-confirmation">
                    <h4>⚠️ Confirm Deletion</h4>
                    <p>Are you sure you want to delete <strong>{member_to_delete.get('name', 'Unknown Member')}</strong>?</p>
                    <p><em>This action cannot be undone.</em></p>
                </div>
                """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("❌ Yes, Delete", type="primary", key="confirm_delete_btn"):
                with st.spinner("Deleting member..."):
                    result = delete_member(member_to_delete['id'])
                    if result:
                        st.success(f"✅ Member '{member_to_delete['name']}' deleted successfully!")
                        st.session_state.delete_confirmation = None
                        time.sleep(1)
                        st.rerun()

        with col2:
            if st.button("↩️ Cancel", key="cancel_delete_btn"):
                st.session_state.delete_confirmation = None
                st.rerun()

        # Main content area based on current state
    try:

        if st.session_state.get("show_add_form"):
            show_add_member_form()
        elif st.session_state.get("edit_member"):
            show_edit_member_form()
        else:
            show_members_list()

    except Exception as e:
        st.error("❌ An error occurred in the main application")
        show_error_details(str(e))

        # Recovery options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        with col2:
            if st.button("🏠 Return to Dashboard"):
                st.session_state.show_add_form = False
                st.session_state.edit_member = None
                st.session_state.show_calendar = False  # <-- Reset calendar state
                st.rerun()

                st.rerun()
    # ✅ Only show if calendar was requested
    if st.session_state.get("show_calendar"):
        st.header("📆 Payment Summary")

        # Select Thursday
        selected_thursday = show_calendar_widget()

        # Fixed weekly amount (or make it user-configurable)
        weekly_amount = 500  # change as needed

        # Show summary only if a Thursday was selected
        if selected_thursday:
            display_payment_summary(selected_thursday, weekly_amount)


def show_dashboard_with_sidebar():
    """Combined function that shows both sidebar and dashboard after login"""
    # Show sidebar first
    show_sidebar()

    # Then show dashboard content
    show_dashboard()

def show_add_member_form():
    st.header("➕ Add Visitors")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.show_add_form = False
            st.rerun()

    with st.form("add_member_form"):
        col1, col2,col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name*", placeholder="Enter member's full name")
            invited_by = st.text_input("Invited By", placeholder="invited by")
            visit_date = st.date_input(
                "Start Date",
                value=date.today(),  # default value is today
                help="Select the start date"  # optional help text
            )

        with col2:
            company_name = st.text_input("Company Name", placeholder="enter the company name")
            fees = st.number_input("Fees", min_value=0, step=100,
                                   help="Must be greater than 0")
        with col3:
            category = st.text_input("Category", placeholder="enter the category")
            payment_mode = st.text_input("Payment Mode", placeholder="e.g., Cash, Online, Cheque")  # ADD THIS LINE

        submitted = st.form_submit_button("Add Visitor", use_container_width=True)

        if submitted:
            with st.spinner("Adding visitor..."):
                result = add_member(name, company_name, category, invited_by, fees, payment_mode,visit_date)  # Now mode is defined
                if result:
                    st.success(f"✅ Member '{name}' added successfully!")
                    time.sleep(1)
                    st.session_state.show_add_form = False
                    st.session_state.show_add_form = False
                    st.rerun()

def show_edit_member_form():
    member_data = st.session_state.edit_member
    st.header(f"✏️ Edit Member: {member_data['name']}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_to_dashboard_btn"):
            st.session_state.edit_member = None
            st.rerun()

    try:
        st.info(
            f"**Member ID:** {member_data.get('id', 'Unknown')} | **Created:** {member_data.get('created_at', 'Unknown')}")
    except Exception as e:
        st.warning("Could not display member metadata")

    with st.form("edit_member_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name*", value=member_data.get('name', ''))
            company_name = st.text_input("Company name", value=member_data.get('company_name', ''))
            category = st.text_input("Category", value=member_data.get('category', ''))
            invited_by = st.text_input("Invited by", value=member_data.get('invited_by', ''))

            fees = st.number_input("Fees*", min_value=0.01, step=0.01,
                                      value=member_data.get('fees', 0), format="%.2f")

        with col2:
            payment_mode = st.text_input("Mode of payment", value=member_data.get('payment_mode', ''))  # ADD THIS LINE

        submitted = st.form_submit_button("Update Member", use_container_width=True)

        if submitted:
            with st.spinner("Updating member..."):
                # FIXED: Correct parameter order
                result = update_member(id, name, company_name, category, invited_by, fees, payment_mode)
                if result:
                    st.success(f"✅ Member '{name}' updated successfully!")
                    time.sleep(1)
                    st.session_state.edit_member = None
                    st.rerun()

def get_thursdays_in_month(year, month):
    """Return all Thursdays in a given month."""
    thursdays = []
    cal = calendar.monthcalendar(year, month)

    for week in cal:
        if week[3] != 0:  # Index 3 = Thursday
            thursdays.append(date(year, month, week[3]))

    return thursdays


def show_calendar_widget():
    st.subheader("📅 Select a Thursday from a Month")

    today = date.today()
    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox("Select Year", [today.year - 1, today.year, today.year + 1], index=1)

    with col2:
        selected_month = st.selectbox(
            "Select Month",
            list(range(1, 13)),
            format_func=lambda m: calendar.month_name[m]
        )

    thursdays = get_thursdays_in_month(selected_year, selected_month)

    if not thursdays:
        st.warning("❌ No Thursdays found in this month.")
        return None

    # Add a placeholder first option to prevent auto-selection
    thursday_options = ["-- Select a Thursday --"] + thursdays

    selected_thursday = st.selectbox(
        "Select a Thursday",
        thursday_options,
        format_func=lambda d: d.strftime("%B %d, %Y") if isinstance(d, date) else d
    )

    # Only proceed if a valid date (not placeholder) is selected
    if isinstance(selected_thursday, date):
        st.session_state.selected_thursday = selected_thursday
        st.success(f"✅ Selected Thursday: {selected_thursday.strftime('%B %d, %Y')}")

        # Load all members if not already loaded or incorrectly typed
        if 'individuals_data' not in st.session_state or not isinstance(st.session_state.individuals_data, list):
            all_members = get_all_members()

            # Ensure it's a list of dicts
            if isinstance(all_members, pd.DataFrame):
                all_members = all_members.to_dict(orient="records")

            selected_date_str = selected_thursday.strftime("%Y-%m-%d")

            # Filter members for selected date
            filtered_members = [
                m for m in all_members
                if str(m.get("visit_date", "")).startswith(selected_date_str)
                            ]

            st.session_state.individuals_data = filtered_members

        return selected_thursday

    # If no valid Thursday selected, clear old data
    st.session_state.individuals_data = []
    return None


def show_members_list():
    # Get members data with error handling
    try:
        if st.session_state.search_query:
            df = search_members(st.session_state.search_query)
            if df is not None:
                st.subheader(f"🔍 Search Results for: '{st.session_state.search_query}'")
            else:
                df = pd.DataFrame()
        else:
            df = get_all_members()
            if df is not None:
                st.subheader("👥 All Members")
            else:
                df = pd.DataFrame()

    except Exception as e:
        st.error("Failed to load members data")
        show_error_details(str(e))
        return

    # Show statistics
    stats = get_stats(df)

    col1, col2, col3 = st.columns(3)

    # Show members table with error handling
    if df is not None and not df.empty:
        try:
            # Create display dataframe with error handling
            display_df = df.copy()

            # Safe payment formatting
            if 'fees' in display_df.columns:
                display_df['fees'] = display_df['fees'].apply(
                    lambda x: f"₹{(x):,.2f}" if pd.notnull(x) else "₹0.00"
                )

            # Safe date formatting
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(
                    display_df['created_at'], errors='coerce'
                ).dt.strftime('%Y-%m-%d')

            # Show table headers
            st.markdown("---")
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1, 1.5, 1])

            with col1:
                st.markdown("**Name**")
            with col2:
                st.markdown("**Company Name**")
            with col3:
                st.markdown("**Category**")
            with col4:
                st.markdown("**Invited by**")
            with col5:
                st.markdown("**Fees**")
            with col6:
                st.markdown("**Payment Mode**")

            st.markdown("---")

            # Show table with error boundaries for each row
            for idx, row in df.iterrows():
                try:
                    with st.container():
                        col1, col2, col3, col4, col5, col6,col7 = st.columns([2, 1.5, 1, 1, 1.5, 1,1])

                        with col1:
                            st.write(f"**{row.get('name', 'Unknown')}**")
                            st.caption(f"ID: {row.get('id', 'Unknown')}")

                        with col2:
                            st.write(row.get('company_name', 'Not specified'))

                        with col3:
                            st.write(row.get('category', 'Not specified'))

                        with col4:
                            st.write(row.get('invited_by', 'Not specified'))

                        with col5:
                            fees_val = row.get('fees', 0)
                            try:
                                st.write(f"₹{(fees_val):,.2f}")
                            except (ValueError, TypeError):
                                st.write("₹0.00")

                        with col6:
                            #mode = row.get('payment_mode', 'Unknown')
                            st.write(row.get('payment_mode', 'Unknown'))

                        with col7:
                            col_edit, col_delete = st.columns(2)

                            with col_edit:
                                if st.button("✏️", key=f"edit_{row['id']}", help="Edit member"):
                                    st.session_state.edit_member = row.to_dict()
                                    st.rerun()

                            with col_delete:
                                if st.button("🗑️", key=f"delete_{row['id']}", help="Delete member"):
                                    st.session_state.delete_confirmation = row.to_dict()
                                    st.rerun()

                        st.divider()

                except Exception as e:
                    st.error(f"Error displaying member row: {str(e)}")
                    logger.error(f"Row display error for member {row.get('id', 'unknown')}: {e}")
                    continue

        except Exception as e:
            st.error("Error displaying members table")
            show_error_details(str(e))

    else:
        if st.session_state.search_query:
            st.info(f"No members found matching '{st.session_state.search_query}'")
            if st.button("🔄 Try Different Search", use_container_width=True):
                st.session_state.search_query = ""
                st.rerun()
        else:
            st.info("No members found. Add some members to get started!")
            if st.button("➕ Add First Member"):
                st.session_state.show_add_form = True
                st.rerun()

def main():
    init_session_state()
    show_dashboard_with_sidebar()

if __name__ == "__main__":
    main()
    init_session_state()