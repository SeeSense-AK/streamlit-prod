# SeeSense Safety Analytics Platform - Production Version

A production-ready cycling safety analytics dashboard built with Streamlit, designed to process real-world cycling data and provide actionable insights for infrastructure planning and safety improvements.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd seesense-dashboard

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The dashboard requires four CSV files in the `data/raw/` directory:

```
data/raw/
├── routes.csv              # Route data with popularity metrics
├── braking_hotspots.csv    # Sudden braking incident locations
├── swerving_hotspots.csv   # Swerving incident locations
└── time_series.csv         # Daily aggregated cycling data
```

### 3. Run the Dashboard

```bash
streamlit run app/main.py
```

The dashboard will automatically guide you through data setup if files are missing.

## 📊 Data Requirements

### Routes Data (`routes.csv`)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| route_id | string | Yes | Unique route identifier |
| start_lat | float | Yes | Starting latitude |
| start_lon | float | Yes | Starting longitude |
| end_lat | float | Yes | Ending latitude |
| end_lon | float | Yes | Ending longitude |
| distinct_cyclists | int | Yes | Number of unique cyclists |
| days_active | int | Yes | Days route has been active |
| popularity_rating | int | Yes | Rating 1-10 |
| avg_speed | float | Yes | Average speed (km/h) |
| avg_duration | float | Yes | Average duration (minutes) |
| route_type | string | Yes | Commute/Leisure/Exercise/Mixed |
| has_bike_lane | boolean | Yes | Whether route has bike lane |
| distance_km | float | No | Route distance (auto-calculated if missing) |

### Braking Hotspots (`braking_hotspots.csv`)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| hotspot_id | string | Yes | Unique hotspot identifier |
| lat | float | Yes | Latitude of hotspot |
| lon | float | Yes | Longitude of hotspot |
| intensity | float | Yes | Intensity score 0-10 |
| incidents_count | int | Yes | Number of braking incidents |
| avg_deceleration | float | Yes | Average deceleration (m/s²) |
| road_type | string | Yes | Junction/Crossing/Roundabout/Straight |
| date_recorded | datetime | Yes | Date recorded (YYYY-MM-DD) |
| surface_quality | string | No | Road surface quality |
| severity_score | float | No | Calculated severity (auto-generated) |

### Swerving Hotspots (`swerving_hotspots.csv`)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| hotspot_id | string | Yes | Unique hotspot identifier |
| lat | float | Yes | Latitude of hotspot |
| lon | float | Yes | Longitude of hotspot |
| intensity | float | Yes | Intensity score 0-10 |
| incidents_count | int | Yes | Number of swerving incidents |
| avg_lateral_movement | float | Yes | Average lateral movement (meters) |
| road_type | string | Yes | Junction/Crossing/Roundabout/Straight |
| date_recorded | datetime | Yes | Date recorded (YYYY-MM-DD) |
| obstruction_present | string | No | Yes/No/Unknown |
| cause_category | string | No | Categorized cause |

### Time Series Data (`time_series.csv`)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| date | datetime | Yes | Date (YYYY-MM-DD) |
| total_rides | int | Yes | Total rides for the day |
| incidents | int | Yes | Safety incidents count |
| avg_speed | float | Yes | Average speed (km/h) |
| avg_braking_events | float | Yes | Average braking events per ride |
| avg_swerving_events | float | Yes | Average swerving events per ride |
| precipitation_mm | float | No | Daily precipitation (mm) |
| temperature | float | No | Temperature (°C) |
| wind_speed | float | No | Wind speed (km/h) |
| visibility_km | float | No | Visibility (km) |

## 🏗️ Project Structure

```
seesense-dashboard/
├── app/
│   ├── main.py                    # Main application entry point
│   ├── components/                # Reusable UI components
│   ├── pages/                     # Dashboard pages/tabs
│   │   └── data_setup.py         # Data setup and validation page
│   ├── core/                     # Core business logic
│   │   └── data_processor.py     # Data loading and processing
│   └── utils/                    # Utility functions
│       ├── config.py             # Configuration management
│       ├── validators.py         # Data validation
│       └── cache.py              # Caching utilities
├── assets/                       # Static assets
│   ├── logo.png
│   └── styles/
│       └── custom.css
├── config/                      # Configuration files
│   ├── settings.yaml           # App settings
│   └── data_schema.yaml        # Data validation schemas
├── data/
│   ├── raw/                    # Raw CSV input files
│   └── processed/              # Processed data cache
├── .devcontainer/              # Development container config
├── requirements.txt
└── README.md
```

## 🛠️ Configuration

### Application Settings (`config/settings.yaml`)

Customize the dashboard by editing the configuration file:

```yaml
app:
  title: "SeeSense Safety Analytics Platform"
  icon: "🚲"
  layout: "wide"

data:
  cache_ttl: 3600  # Cache duration in seconds
  max_file_size_mb: 100

visualization:
  default_map_center:
    lat: 51.5074  # London coordinates
    lon: -0.1278
  default_zoom: 12
```

### Environment Variables

Set environment variables for production deployment:

```bash
export DASHBOARD_ENV=production
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🚀 Deployment Options

### Local Deployment

```bash
streamlit run app/main.py --server.port 8501
```

### Docker Deployment

```bash
# Build the container
docker build -t seesense-dashboard .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data seesense-dashboard
```

### Streamlit Cloud

1. Push your repository to GitHub
2. Connect to Streamlit Cloud
3. Add your CSV files to the `data/raw/` directory
4. Deploy!

## 📁 Data Management

### Adding New Data

1. Place CSV files in `data/raw/` directory
2. Ensure files match the required schema
3. Use the "Data Setup" page to validate and load data
4. Dashboard will automatically process and cache the data

### Data Validation

The dashboard includes comprehensive data validation:

- **Schema validation**: Checks required columns and data types
- **Constraint validation**: Validates value ranges and categories  
- **Quality checks**: Identifies missing values and outliers
- **Duplicate detection**: Finds duplicate rows and IDs

### Data Processing Pipeline

1. **Load**: Read CSV files from `data/raw/`
2. **Validate**: Check schema and data quality
3. **Clean**: Remove invalid data and standardize formats
4. **Process**: Calculate derived metrics and features
5. **Cache**: Store processed data for performance

## 🔧 Development

### Adding New Features

1. **New Pages**: Add to `app/pages/`
2. **Components**: Add reusable UI components to `app/components/`
3. **Analytics**: Add new analysis functions to `app/core/`
4. **Configuration**: Update `config/settings.yaml` for new settings

### Data Schema Updates

1. Update `config/data_schema.yaml` with new requirements
2. Modify validation logic in `app/utils/validators.py`
3. Update processing functions in `app/core/data_processor.py`

### Testing Your Data

Use the built-in data validation tools:

1. Go to "Data Setup" page
2. Upload or validate existing CSV files
3. Review validation results and fix any issues
4. Download template files for reference

## 📊 Dashboard Features

### Current Features (Phase 1)

- ✅ **Data Setup & Validation**: Comprehensive CSV validation and setup guidance
- ✅ **Data Processing Pipeline**: Automated data cleaning and feature engineering
- ✅ **Configuration Management**: Flexible YAML-based configuration
- ✅ **Caching System**: Intelligent data caching for performance
- ✅ **Error Handling**: Graceful handling of missing or invalid data

### Planned Features (Phase 2)

- 🔄 **Dashboard Overview**: Key safety metrics and trends
- 🔄 **ML Insights**: Predictive analytics and risk modeling
- 🔄 **Spatial Analysis**: Interactive maps and hotspot analysis
- 🔄 **Advanced Analytics**: Time series analysis and anomaly detection
- 🔄 **Actionable Insights**: AI-generated recommendations

## 🆘 Troubleshooting

### Common Issues

**"No datasets available"**
- Ensure CSV files are in `data/raw/` directory
- Check file names match exactly: `routes.csv`, `braking_hotspots.csv`, etc.
- Use "Data Setup" page to validate file formats

**"Validation failed"**
- Review error messages in "Data Setup" page
- Check data types match schema requirements
- Ensure required columns are present

**"Dashboard slow to load"**
- Large datasets may take time to process initially
- Data is cached after first load for better performance
- Consider filtering data or reducing file sizes

### Getting Help

1. Check the "Data Setup" page for validation errors
2. Review the sample data formats provided
3. Check log files in `logs/dashboard.log`
4. Contact support: support@seesense.cc
