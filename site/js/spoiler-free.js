/**
 * Spoiler-Free Manager for Sumo Companion App
 *
 * Handles the spoiler-free viewing mechanism:
 * - Remembers user's selected tournament and last watched day
 * - Shows/hides content based on selection
 * - Persists state in localStorage
 */

class SpoilerFreeManager {
    constructor() {
        this.storageKey = 'sumo-companion-state';
        this.state = this.loadState();
        this.init();
    }

    loadState() {
        try {
            const stored = localStorage.getItem(this.storageKey);
            return stored ? JSON.parse(stored) : {
                selectedBasho: null,
                lastDayWatched: 0
            };
        } catch (e) {
            console.error('Error loading state:', e);
            return { selectedBasho: null, lastDayWatched: 0 };
        }
    }

    saveState() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.state));
        } catch (e) {
            console.error('Error saving state:', e);
        }
    }

    init() {
        // Only run on pages with the controls
        const bashoSelector = document.getElementById('basho-selector');
        const daySelector = document.getElementById('day-selector');

        if (bashoSelector) {
            this.bindBashoSelector(bashoSelector);
        }

        if (daySelector) {
            this.bindDaySelector(daySelector);
        }

        // Check for spoiler elements on any page
        this.updateSpoilerVisibility();
    }

    bindBashoSelector(selector) {
        // Set initial value
        if (this.state.selectedBasho) {
            selector.value = this.state.selectedBasho;
        }

        selector.addEventListener('change', (e) => {
            this.state.selectedBasho = e.target.value;
            this.state.lastDayWatched = 0; // Reset on basho change
            this.saveState();
            this.updateSpoilerVisibility();
        });
    }

    bindDaySelector(selector) {
        // Set initial value
        if (this.state.lastDayWatched !== undefined) {
            selector.value = this.state.lastDayWatched;
        }

        selector.addEventListener('change', (e) => {
            this.state.lastDayWatched = parseInt(e.target.value);
            this.saveState();
            this.updateSpoilerVisibility();
        });
    }

    updateSpoilerVisibility() {
        const { selectedBasho, lastDayWatched } = this.state;

        // Update all elements with data-day attribute
        const dayElements = document.querySelectorAll('[data-day]');

        dayElements.forEach(el => {
            const day = parseInt(el.dataset.day);
            const type = el.dataset.type; // 'preview' or 'results'

            if (day <= lastDayWatched) {
                // Show both preview and results for watched days
                el.classList.remove('spoiler-hidden', 'hidden');
                el.classList.add('visible');
            } else if (day === lastDayWatched + 1) {
                // Show preview only for next day
                if (type === 'preview') {
                    el.classList.remove('spoiler-hidden', 'hidden');
                    el.classList.add('visible');
                } else {
                    el.classList.add('spoiler-hidden');
                    el.classList.remove('visible');
                }
            } else {
                // Hide all for future days
                el.classList.add('hidden');
                el.classList.remove('visible', 'spoiler-hidden');
            }
        });

        // Update yusho race visibility
        const yushoSection = document.getElementById('yusho-race');
        if (yushoSection) {
            const standings = yushoSection.querySelectorAll('[data-through-day]');
            standings.forEach(el => {
                const throughDay = parseInt(el.dataset.throughDay);
                if (throughDay > lastDayWatched) {
                    el.classList.add('hidden');
                } else {
                    el.classList.remove('hidden');
                }
            });
        }
    }

    setBasho(bashoId) {
        this.state.selectedBasho = bashoId;
        this.saveState();
        this.updateSpoilerVisibility();
    }

    setLastDayWatched(day) {
        this.state.lastDayWatched = day;
        this.saveState();
        this.updateSpoilerVisibility();
    }

    getState() {
        return { ...this.state };
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.spoilerFree = new SpoilerFreeManager();
});

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpoilerFreeManager;
}
