/**
 * Frontend JavaScript for Hallucination Detection System
 * Powered by Qwen 2.5 (1.5B) with 18+ Knowledge APIs
 */

// Backend runs on 8000, frontend served by Live Server on 5500
const API_BASE_URL = 'http://localhost:8000';

/**
 * Set a question in the input field
 */
function setQuestion(question) {
    document.getElementById('questionInput').value = question;
}

/**
 * Main function to ask a question and display results
 */
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    hideResults();
    
    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}\n\nMake sure the backend server is running at ${API_BASE_URL}`);
    } finally {
        setLoadingState(false);
    }
}

/**
 * Set loading state for the button
 */
function setLoadingState(isLoading) {
    const button = document.getElementById('askButton');
    const buttonText = document.getElementById('buttonText');
    const spinner = document.getElementById('loadingSpinner');
    
    button.disabled = isLoading;
    buttonText.style.display = isLoading ? 'none' : 'inline';
    spinner.style.display = isLoading ? 'inline' : 'none';
}

/**
 * Hide results section
 */
function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

/**
 * Display API results in the UI
 */
function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }, 100);
    
    // Display hallucination alert
    displayHallucinationAlert(data.is_hallucinated, data.hallucination_score);
    
    // Display score
    displayScore(data.hallucination_score);
    
    // Display answers
    displayOriginalAnswer(data.answer, data.claims, data.claim_scores);
    displayCorrectedAnswer(data.corrected_answer);
    
    // Display evidence
    displayEvidence(data.evidence);
    
    // Display technical details
    displayTechnicalDetails(data);
}

/**
 * Display hallucination alert banner
 */
function displayHallucinationAlert(isHallucinated, score) {
    const alert = document.getElementById('hallucinationAlert');
    
    if (isHallucinated) {
        alert.className = 'alert hallucinated';
        alert.innerHTML = `
            <span style="font-size: 1.5rem;">⚠️</span>
            <span>Hallucination Detected! The answer may contain unverified claims. See corrected version below.</span>
        `;
    } else if (score > 0.3) {
        alert.className = 'alert uncertain';
        alert.innerHTML = `
            <span style="font-size: 1.5rem;">⚡</span>
            <span>Moderate Confidence: Some claims may need additional verification.</span>
        `;
    } else {
        alert.className = 'alert verified';
        alert.innerHTML = `
            <span style="font-size: 1.5rem;">✅</span>
            <span>Answer Verified! Claims are well-supported by evidence.</span>
        `;
    }
}

/**
 * Display hallucination score with visual indicator
 */
function displayScore(score) {
    const scoreValue = document.getElementById('scoreValue');
    const scoreBar = document.getElementById('scoreBar');
    
    // Update score value
    scoreValue.textContent = score.toFixed(3);
    
    // Update color based on score
    if (score > 0.45) {
        scoreValue.style.color = '#ef4444'; // Red
    } else if (score > 0.3) {
        scoreValue.style.color = '#f59e0b'; // Orange
    } else {
        scoreValue.style.color = '#10b981'; // Green
    }
    
    // Update score bar (inverse - lower score is better)
    const barWidth = (1 - score) * 100;
    scoreBar.style.width = `${barWidth}%`;
}

/**
 * Display original generated answer
 */
function displayOriginalAnswer(answer, claims, claimScores) {
    const answerDiv = document.getElementById('originalAnswer');
    answerDiv.textContent = answer;
    
    // Display claims breakdown
    if (claims && claims.length > 0) {
        const claimsList = document.getElementById('claimsList');
        claimsList.innerHTML = '<h4 style="margin-top: 15px; margin-bottom: 10px;">📝 Claim Analysis</h4>';
        
        claims.forEach((claim, index) => {
            const score = claimScores[index];
            const isVerified = score > 0.5;
            
            const claimItem = document.createElement('div');
            claimItem.className = `claim-item ${isVerified ? 'verified' : 'unverified'}`;
            claimItem.innerHTML = `
                <span class="claim-score">${(score * 100).toFixed(0)}%</span>
                ${claim}
            `;
            
            claimsList.appendChild(claimItem);
        });
    }
}

/**
 * Display corrected answer if available
 */
function displayCorrectedAnswer(correctedAnswer) {
    const correctedCard = document.getElementById('correctedCard');
    const correctedDiv = document.getElementById('correctedAnswer');
    
    if (correctedAnswer) {
        correctedCard.style.display = 'block';
        correctedDiv.textContent = correctedAnswer;
    } else {
        correctedCard.style.display = 'none';
    }
}

/**
 * Display retrieved evidence passages
 */
function displayEvidence(evidence) {
    const evidenceList = document.getElementById('evidenceList');
    evidenceList.innerHTML = '';
    
    if (!evidence || evidence.length === 0) {
        evidenceList.innerHTML = '<p style="color: #64748b;">No evidence retrieved.</p>';
        return;
    }
    
    // Add API summary
    const apiSources = [...new Set(evidence.map(e => {
        const source = e.source.split(':')[0];
        return source;
    }))];
    
    const summaryDiv = document.createElement('div');
    summaryDiv.style.cssText = 'margin-bottom: 15px; padding: 10px; background: #f1f5f9; border-radius: 8px; font-size: 0.9rem;';
    summaryDiv.innerHTML = `
        <strong>📡 Sources Consulted:</strong> ${apiSources.join(', ')} 
        <span style="color: #64748b;">(${evidence.length} evidence passages)</span>
    `;
    evidenceList.appendChild(summaryDiv);
    
    evidence.forEach((item, index) => {
        const evidenceItem = document.createElement('div');
        evidenceItem.className = 'evidence-item';
        
        // Add API icon based on source
        const sourceIcon = getSourceIcon(item.source);
        
        evidenceItem.innerHTML = `
            <div class="evidence-source">${sourceIcon} Source ${index + 1}: ${item.source}</div>
            <div class="evidence-snippet">${item.snippet}</div>
            ${item.score ? `<div class="evidence-score">Relevance: ${(item.score * 100).toFixed(1)}%</div>` : ''}
        `;
        
        evidenceList.appendChild(evidenceItem);
    });
}

/**
 * Get icon for evidence source
 */
function getSourceIcon(source) {
    const sourceLower = source.toLowerCase();
    if (sourceLower.includes('wikipedia')) return '📖';
    if (sourceLower.includes('dbpedia')) return '🗂️';
    if (sourceLower.includes('wikidata')) return '🔗';
    if (sourceLower.includes('stack')) return '💻';
    if (sourceLower.includes('arxiv')) return '📄';
    if (sourceLower.includes('scholar')) return '🎓';
    if (sourceLower.includes('nasa')) return '🚀';
    if (sourceLower.includes('pubchem')) return '🧪';
    if (sourceLower.includes('news')) return '📰';
    if (sourceLower.includes('library')) return '📚';
    if (sourceLower.includes('countries')) return '🌍';
    if (sourceLower.includes('world bank')) return '💰';
    if (sourceLower.includes('duckduckgo')) return '🔍';
    return '📌';
}

/**
 * Display technical details
 */
function displayTechnicalDetails(data) {
    const technicalInfo = document.getElementById('technicalInfo');
    
    const details = {
        'Model': 'Qwen 2.5 (1.5B parameters)',
        'Embedding Model': 'MiniLM-L6 (384-dim)',
        'Hallucination Score': data.hallucination_score.toFixed(4),
        'Is Hallucinated': data.is_hallucinated ? 'Yes ⚠️' : 'No ✅',
        'Number of Claims': data.claims?.length || 0,
        'Evidence Retrieved': data.evidence?.length || 0,
        'Has Correction': data.corrected_answer ? 'Yes' : 'No'
    };
    
    if (data.claim_scores && data.claim_scores.length > 0) {
        details['Mean Claim Support'] = (data.claim_scores.reduce((a, b) => a + b, 0) / data.claim_scores.length).toFixed(4);
        details['Min Claim Score'] = Math.min(...data.claim_scores).toFixed(4);
        details['Max Claim Score'] = Math.max(...data.claim_scores).toFixed(4);
    }
    
    // Count unique API sources
    if (data.evidence && data.evidence.length > 0) {
        const apiSources = [...new Set(data.evidence.map(e => e.source.split(':')[0]))];
        details['API Sources Used'] = `${apiSources.length} APIs`;
    }
    
    let html = '<pre>';
    for (const [key, value] of Object.entries(details)) {
        html += `${key.padEnd(25)}: ${value}\n`;
    }
    html += '</pre>';
    
    technicalInfo.innerHTML = html;
}

/**
 * Handle Enter key in textarea
 */
document.addEventListener('DOMContentLoaded', function() {
    const questionInput = document.getElementById('questionInput');
    
    questionInput.addEventListener('keydown', function(event) {
        // Submit on Ctrl+Enter or Cmd+Enter
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            askQuestion();
        }
    });
});
