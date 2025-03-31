
const svgMatrixD3 = d3.select("svg#matrix");
const svgMatrixElem = document.getElementById("matrix");

function scale_svg_matrix(number_nodes) {
    const totalwidth = 53 * (number_nodes + 1) + 40;
    const totalheight = 51 * (number_nodes + 1);

    // JavaScript code to scale the SVG if width is larger
    const svgMaxWidth =
        svgMatrixElem.parentElement.parentElement.clientWidth;
    const svgClientWidth = svgMatrixElem.clientWidth;
    const scaleRatio = svgMaxWidth / svgClientWidth;
    if (scaleRatio < 1) {
        svgMatrixD3.style("transform", `scale(${scaleRatio})`);
        svgMatrixD3.style("transform-origin", "top left");
        svgMatrixElem.parentElement.style.height = `${totalheight * scaleRatio
        }px`;
        svgMatrixElem.parentElement.style.width = `${totalwidth * scaleRatio
        }px`;
    }
}

function createPermutationMatrix(indices) {
    svgMatrixD3.selectAll("*").remove();
    // Define the size of the matrix based on the length of indices array
    const matrixSize = indices.length;

    // Create the SVG container
    svgMatrixD3
        .attr("width", matrixSize * 50 + 50)
        .attr("height", matrixSize * 50 + 50)
        .attr("style", "background-color: #f2f2f2;");

    // Create a matrix of rectangles
    const matrix = svgMatrixD3
        .selectAll("rect")
        .data(indices)
        .enter()
        .append("rect")
        .attr("style", "fill: #333; stroke: black; stroke-width: 1px;")
        .attr("x", (d, i) => i * 50 + 50)
        .attr("y", (d) => d * 50 + 50)
        .attr("width", 50)
        .attr("height", 50);

    // Add text labels to the matrix cells
    const labels = svgMatrixD3
        .selectAll("text")
        .data(indices)
        .enter()
        .append("text")
        .attr(
            "style",
            "font-family: Arial, sans-serif; font-size: 12px; fill: #fff;"
        )
        .attr("x", (d, i) => i * 50 + 75)
        .attr("y", (d) => d * 50 + 75)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((i, j) => `${i},${j}`);

    // Add x-axis labels
    const xAxisLabels = svgMatrixD3
        .selectAll(".x-axis-label")
        .data(indices)
        .enter()
        .append("text")
        .attr(
            "style",
            "font-family: Arial, sans-serif; font-size: 12px; fill: #333;"
        )
        .attr("x", (d, i) => i * 50 + 75)
        .attr("y", 25)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d, i) => i);

    // Add y-axis labels
    const yAxisLabels = svgMatrixD3
        .selectAll(".y-axis-label")
        .data(indices)
        .enter()
        .append("text")
        .attr(
            "style",
            "font-family: Arial, sans-serif; font-size: 12px; fill: #333;"
        )
        .attr("x", 25)
        .attr("y", (d, i) => i * 50 + 75)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d, i) => i);

    // Add horizontal grid lines
    const horizontalLines = svgMatrixD3
        .selectAll(".horizontal-line")
        .data(indices)
        .enter()
        .append("line")
        .attr("style", "stroke: #ccc; stroke-width: 1px;")
        .attr("x1", 50)
        .attr("y1", (d, i) => i * 50 + 50)
        .attr("x2", matrixSize * 50 + 50)
        .attr("y2", (d, i) => i * 50 + 50)
        .attr("class", "horizontal-line");

    // Add vertical grid lines
    const verticalLines = svgMatrixD3
        .selectAll(".vertical-line")
        .data(indices)
        .enter()
        .append("line")
        .attr("style", "stroke: #ccc; stroke-width: 1px;")
        .attr("x1", (d, i) => i * 50 + 50)
        .attr("y1", 50)
        .attr("x2", (d, i) => i * 50 + 50)
        .attr("y2", matrixSize * 50 + 50)
        .attr("class", "vertical-line");

    scale_svg_matrix(matrixSize);
}

// Example usage with an array of indices

function createIndices(n) {
    let omega = [];
    omega[0] = n - 1;

    for (let i = 0; 2 * i + 1 < n; i++) {
        omega[2 * i] = Math.floor((omega[i] + omega[0]) / 2);
        omega[2 * i + 1] = Math.floor(omega[i] / 2);
    }

    return omega;
}

function isPowerTwo(num) {
    return num > 0 && Math.abs(2 ** Math.round(Math.log2(num)) - num) < 1e-3;
}

const inputMatrix = document.getElementById("inputMatrix");

function onApplyMatrixBtn() {
    const numnodes = parseInt(inputMatrix.value);
    if (isPowerTwo(numnodes)) {
        createPermutationMatrix(createIndices(numnodes));
    } else {
        alert("Enter power of two");
    }
}

inputMatrix.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        onApplyMatrixBtn();
    }
});

function onSaveMatrixBtn() {
    saveSvg(svgMatrixElem.cloneNode(true), "matrix.svg");
}

onApplyMatrixBtn();