// frontend/static/js/map.js

// init map
let map = L.map('map').setView([31.7917, -7.0926], 6);  // Centered on Morocco

// Add base tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Layer groups to manage visibility
const layers = {
    rmaOffices: L.layerGroup().addTo(map),
    rmaConnections: L.layerGroup().addTo(map),
    banks: L.layerGroup(),
    competitors: L.layerGroup(),
    population: L.layerGroup().addTo(map),
    coverageScores: L.layerGroup().addTo(map)
};

// Layer control
const overlays = {
    "RMA Offices": layers.rmaOffices,
    "RMA Connections": layers.rmaConnections,
    "Banks": layers.banks,
    "Competitors": layers.competitors,
    "Population": layers.population,
    "Coverage Scores": layers.coverageScores
};

L.control.layers(null, overlays).addTo(map);

// Style functions
const officeStyle = (feature) => {
    return {
        radius: 8,
        fillColor: getOfficeColor(feature.properties.office_type),
        color: "#000",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
    };
};

const getOfficeColor = (type) => {
    switch(type) {
        case 'HEADQUARTERS': return '#e41a1c';
        case 'REGIONAL': return '#377eb8';
        case 'LOCAL': return '#4daf4a';
        case 'SERVICE_CENTER': return '#984ea3';
        default: return '#ff7f00';
    }
};

const connectionStyle = (feature) => {
    return {
        color: "#3388ff",
        weight: feature.properties.weight,
        opacity: 0.65,
        dashArray: feature.properties.connection_type === 'VIRTUAL' ? '5, 5' : null
    };
};

const bankStyle = (feature) => {
    return {
        radius: 6,
        fillColor: feature.properties.is_partner ? '#1e90ff' : '#a9a9a9',
        color: "#000",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.7
    };
};

const competitorStyle = (feature) => {
    return {
        radius: 6,
        fillColor: '#ff4500',
        color: "#000",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.7
    };
};

// Population choropleth style
const populationStyle = (feature) => {
    return {
        fillColor: getPopulationColor(feature.properties.population_density),
        weight: 1,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.7
    };
};

const getPopulationColor = (density) => {
    return density > 1000 ? '#800026' :
           density > 500  ? '#BD0026' :
           density > 200  ? '#E31A1C' :
           density > 100  ? '#FC4E2A' :
           density > 50   ? '#FD8D3C' :
           density > 20   ? '#FEB24C' :
           density > 10   ? '#FED976' : '#FFEDA0';
};

// Coverage score style
const coverageStyle = (feature) => {
    return {
        fillColor: getCoverageColor(feature.properties.score),
        weight: 1,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.7
    };
};

const getCoverageColor = (score) => {
    return score > 90 ? '#006837' :
           score > 75 ? '#31a354' :
           score > 60 ? '#78c679' :
           score > 45 ? '#c2e699' :
           score > 30 ? '#ffffcc' :
           score > 15 ? '#fdae61' : '#d73027';
};

// Popup functions
const officePopup = (feature, layer) => {
    if (feature.properties) {
        layer.bindPopup(`
            <strong>${feature.properties.name}</strong><br>
            Type: ${feature.properties.office_type}<br>
            Address: ${feature.properties.address}<br>
            ${feature.properties.city}, ${feature.properties.state} ${feature.properties.zipcode}<br>
            Service Radius: ${feature.properties.service_radius_km} km
        `);
    }
};

const bankPopup = (feature, layer) => {
    if (feature.properties) {
        layer.bindPopup(`
            <strong>${feature.properties.institution_name}</strong><br>
            ${feature.properties.branch_name ? feature.properties.branch_name + '<br>' : ''}
            Address: ${feature.properties.address}<br>
            ${feature.properties.city}, ${feature.properties.state} ${feature.properties.zipcode}<br>
            ${feature.properties.is_partner ? '<strong>Partner Bank</strong>' : 'Non-partner bank'}
        `);
    }
};

const competitorPopup = (feature, layer) => {
    if (feature.properties) {
        layer.bindPopup(`
            <strong>${feature.properties.company_name}</strong><br>
            Type: ${feature.properties.competitor_type}<br>
            Address: ${feature.properties.address}<br>
            ${feature.properties.city}, ${feature.properties.state} ${feature.properties.zipcode}<br>
            ${feature.properties.market_share ? 'Market Share: ' + feature.properties.market_share + '%' : ''}
        `);
    }
};

const populationPopup = (feature, layer) => {
    if (feature.properties) {
        layer.bindPopup(`
            <strong>${feature.properties.name}</strong><br>
            Population: ${feature.properties.total_population.toLocaleString()}<br>
            Density: ${feature.properties.population_density.toFixed(2)} per km²<br>
            ${feature.properties.median_income ? 'Median Income: $' + feature.properties.median_income.toLocaleString() : ''}<br>
            ${feature.properties.median_age ? 'Median Age: ' + feature.properties.median_age : ''}
        `);
    }
};

const coveragePopup = (feature, layer) => {
    if (feature.properties) {
        layer.bindPopup(`
            <strong>${feature.properties.area_name}</strong><br>
            Coverage Score: <strong>${feature.properties.score.toFixed(1)}/100</strong><br>
            Population Covered: ${feature.properties.population_covered.toLocaleString()} 
            (${feature.properties.coverage_percentage.toFixed(1)}%)<br>
            Nearest Office: ${feature.properties.nearest_office}<br>
            <button onclick="showCoverageDetails(${feature.properties.area_id})">Details</button>
        `);
    }
};

// Load data from API endpoints
function loadAllData() {
    // Load RMA Offices
    fetch('/api/rma-offices/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                pointToLayer: (feature, latlng) => {
                    return L.circleMarker(latlng, officeStyle(feature));
                },
                onEachFeature: officePopup
            }).addTo(layers.rmaOffices);
        });

    // Load RMA Connections
    fetch('/api/rma-connections/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                style: connectionStyle,
                onEachFeature: (feature, layer) => {
                    if (feature.properties) {
                        layer.bindPopup(`
                            <strong>Connection</strong><br>
                            From: ${feature.properties.from_office}<br>
                            To: ${feature.properties.to_office}<br>
                            Type: ${feature.properties.connection_type}<br>
                            Weight: ${feature.properties.weight}
                        `);
                    }
                }
            }).addTo(layers.rmaConnections);
        });

    // Load Banks
    fetch('/api/banks/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                pointToLayer: (feature, latlng) => {
                    return L.circleMarker(latlng, bankStyle(feature));
                },
                onEachFeature: bankPopup
            }).addTo(layers.banks);
        });

    // Load Competitors
    fetch('/api/competitors/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                pointToLayer: (feature, latlng) => {
                    return L.circleMarker(latlng, competitorStyle(feature));
                },
                onEachFeature: competitorPopup
            }).addTo(layers.competitors);
        });

    // Load Population
    fetch('/api/population/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                style: populationStyle,
                onEachFeature: populationPopup
            }).addTo(layers.population);
        });

    // Load Coverage Scores
    fetch('/api/coverage-scores/')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                style: coverageStyle,
                onEachFeature: coveragePopup
            }).addTo(layers.coverageScores);
        });
}

// Coverage Details Modal
function showCoverageDetails(areaId) {
    fetch(`/api/coverage-scores/${areaId}/details/`)
        .then(response => response.json())
        .then(data => {
            // Create and populate modal with coverage details
            const modal = document.getElementById('coverage-modal');
            document.getElementById('coverage-title').textContent = data.area_name;
            
            // Populate detail fields
            document.getElementById('coverage-score').textContent = data.score.toFixed(1);
            document.getElementById('population-covered').textContent = 
                `${data.population_covered.toLocaleString()} (${data.coverage_percentage.toFixed(1)}%)`;
            document.getElementById('nearest-office').textContent = data.nearest_office;
            document.getElementById('competitor-factor').textContent = data.competitor_factor.toFixed(1);
            document.getElementById('bank-factor').textContent = data.bank_partnership_factor.toFixed(1);
            
            // Show the modal
            modal.style.display = 'block';
        });
}

// Initialize the map when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadAllData();
    
    // Add legend for coverage scores
    const coverageLegend = L.control({position: 'bottomright'});
    coverageLegend.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        const grades = [0, 15, 30, 45, 60, 75, 90];
        const labels = [];
        
        div.innerHTML = '<h4>Coverage Score</h4>';
        
        for (let i = 0; i < grades.length; i++) {
            div.innerHTML +=
                '<i style="background:' + getCoverageColor(grades[i] + 1) + '"></i> ' +
                grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
        }
        
        return div;
    };
    coverageLegend.addTo(map);
});