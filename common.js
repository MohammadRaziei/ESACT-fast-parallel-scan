function isPowerTwo(num) {
    return num > 0 && Math.abs(2 ** Math.round(Math.log2(num)) - num) < 1e-3;
  }
  

document.addEventListener('DOMContentLoaded', function() {
    const multiplierInputs = document.querySelectorAll('.multiplier-input');
    
    multiplierInputs.forEach(input => {
        let lastValue = parseInt(input.value) || 16; // مقدار پیش‌فرض 16
        const min = parseInt(input.min) || 2;
        const max = parseInt(input.max) || 1024;
        let isMouseDown = false;


        input.addEventListener('mousedown', function() {
            isMouseDown = true;
            lastValue = parseInt(this.value) || lastValue;
        });

        input.addEventListener('mouseup', function() {
            isMouseDown = false;
        });

        input.addEventListener('input', function() {
            if (!isMouseDown) return;

            const newValue = parseInt(this.value);
            if (isNaN(newValue)) return;

            if (newValue > lastValue) {
                lastValue = Math.min(lastValue * 2, max);
            } else if (newValue < lastValue) {
                lastValue = Math.max(Math.floor(lastValue / 2), min);
            }
            
            this.value = lastValue;
        });
    });
});