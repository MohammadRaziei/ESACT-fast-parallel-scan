
// window.addEventListener('DOMContentLoaded', () => {
const input = document.getElementById("inputNumberOfNodes");

// Your D3.js code to fill the SVG goes here

var svgD3 = d3
.select("svg#canvas")
.attr("preserveAspectRatio", "xMinYMin meet");

const svgElem = document.getElementById("canvas");

// https://www.w3.org/TR/SVG11/types.html#ColorKeywords

function Text(x, y, width, height, mathExpression) {
this.x = x;
this.y = y;
this.mathExpression = mathExpression;
this.width = width;
this.height = height;

if (this.mathExpression !== "") {
// Render the LaTeX expression using KaTeX
const renderedHTML = katex.renderToString(this.mathExpression, {
throwOnError: false,
});

const renderedHTMLcenter = `<div style="display: flex; justify-content: center; align-items: center; height: ${this.height}px; width: ${this.width}px;">${renderedHTML}</div>`;

// Create an SVG text element
this.elem = svgD3
.append("foreignObject")
.attr("x", this.x)
.attr("y", this.y)
.attr("width", this.width) // Adjust the width as needed
.attr("height", this.height) // Adjust the height as needed
.html(renderedHTMLcenter);
}

// Return a reference to the element for future use
return this;
}

// Node component representing a rectangle
class Node {
static width = 100;
static height = 50;

constructor(x, y, colorfill = "powderblue", text = "") {
this.x = x;
this.y = y;
this.colorfill = colorfill;
this.width = Node.width;
this.height = Node.height;
this.text = text;
}

draw() {
this.elem = svgD3
.append("rect")
.attr("x", this.x)
.attr("y", this.y)
.attr("width", this.width)
.attr("height", this.height)
.attr("stroke", "black")
.attr("fill", this.colorfill);

this.addText(this.text);

// Return self for method chaining
return this;
}

addText(text) {
Text(this.x, this.y, this.width, this.height, text);
}

getBottomCenter() {
return { x: this.x + this.width / 2, y: this.y + this.height };
}

getTopCenter() {
return { x: this.x + this.width / 2, y: this.y };
}
}

// Arrow component to draw an arrow between two nodes
class Arrow {
constructor(fromNode, toNode, color = "black") {
this.draw = function () {
var start = fromNode.getBottomCenter();
var end = toNode.getTopCenter();

const calcX = (n1, n2) => {
const tmp = 0.95 * n1.x + 0.05 * n2.x;
return Math.max(
Math.min(tmp, n1.x + Node.width * 0.3),
n1.x - Node.width * 0.3
);
};
svgD3
.append("line")
.attr("x1", calcX(start, end))
.attr("y1", start.y)
.attr("x2", calcX(end, start))
.attr("y2", end.y - 1)
.attr("stroke", color)
.attr("stroke-width", 2)
.attr("marker-end", "url(#arrow)");
};
}
}

function createIndices(n) {
let omega = [];
omega[0] = n - 1;

for (let i = 0; 2 * i + 1 < n; i++) {
omega[2 * i] = Math.floor((omega[i] + omega[0]) / 2);
omega[2 * i + 1] = Math.floor(omega[i] / 2);
}

return omega;
}

function scale_svg(number_nodes) {
const number_of_steps = Math.round(Math.log2(number_nodes) + 1) * 2;
const totalwidth = Node.width * number_nodes + 2;
const totalheight = Node.height * 2 * (number_of_steps + 2.5) + 2;

// JavaScript code to scale the SVG if width is larger
const svgMaxWidth = svgElem.parentElement.clientWidth;
const svgClientWidth = svgElem.clientWidth;
const scaleRatio = svgMaxWidth / svgClientWidth;
if (scaleRatio < 1) {
svgD3.style("transform", `scale(${scaleRatio})`);
svgD3.style("transform-origin", "top left");
svgElem.parentElement.style.height = `${totalheight * scaleRatio}px`;
svgElem.parentElement.style.width = `${totalwidth * scaleRatio}px`;
}
}

function create_svg(number_nodes) {
svgD3.selectAll("*").remove();
svgD3
.append("svg:defs")
.append("svg:marker")
.attr("id", "arrow")
.attr("viewBox", "0 -5 10 10")
.attr("refX", 8) // Adjust the refX to offset the arrow to align properly
.attr("refY", 0)
.attr("markerWidth", 4)
.attr("markerHeight", 4)
.attr("orient", "auto")
.append("svg:path")
.attr("d", "M0,-5L10,0L0,5")
.attr("fill", "black");

const selectAlg = document.getElementById("selectAlgorithms");

const number_of_steps = Math.round(Math.log2(number_nodes) + 1) * 2;

// Create two instances of the Node component

var nodes = Array.from({ length: number_of_steps }, (_, col) =>
Array.from({ length: number_nodes }, (_, row) =>
new Node(row * Node.width, (col + 1.5) * (Node.height * 2)).draw()
)
);

var input_nodes = Array.from({ length: number_nodes }, (_, i) =>
new Node(i * Node.width, 0, "aquamarine", `X_{${i}}`).draw()
);

var output_nodes = Array.from({ length: number_nodes }, (_, i) =>
new Node(
i * Node.width,
(number_of_steps + 2) * (Node.height * 2),
"aquamarine",
`Y_{${i}}`
).draw()
);

const totalwidth = Node.width * number_nodes + 2;
const totalheight = Node.height * 2 * (number_of_steps + 2.5) + 2;

svgElem.setAttribute("width", totalwidth + "px");
svgElem.setAttribute("height", totalheight + "px");

// svgD3.style("width", totalwidth + 'px').style("height", totalheight + 'px');

switch (selectAlg.value) {
case "blelloch":
{
for (let i = 0; i < number_nodes; i++) {
new Arrow(input_nodes[i], nodes[0][i], "darkcyan").draw();
nodes[0][i].addText(`X_{${i}}`);
}

let row = 0;
for (let s = 2; s <= number_nodes; s <<= 1) {
for (let i = 0; i < number_nodes / s; i++) {
new Arrow(
nodes[row][s * i + Math.floor(s / 2) - 1],
nodes[row + 1][s * i + s - 1]
).draw();
new Arrow(
nodes[row][s * i + s - 1],
nodes[row + 1][s * i + s - 1]
).draw();
}
row++;
}
const sum_node = nodes[row++][number_nodes - 1];
new Arrow(sum_node, nodes[row][number_nodes - 1], "red").draw();
sum_node.elem.attr("fill", "lightcyan");
if (number_nodes > 1)
sum_node.addText(`\\sum_{i=0}^{${number_nodes - 1}}X_i`);
nodes[row][number_nodes - 1].addText("0");

for (let s = number_nodes; s > 1; s >>= 1) {
for (let i = 0; i < number_nodes / s; i++) {
new Arrow(
nodes[row][s * i + Math.floor(s / 2) - 1],
nodes[row + 1][s * i + s - 1]
).draw();
new Arrow(
nodes[row][s * i + s - 1],
nodes[row + 1][s * i + s - 1]
).draw();
new Arrow(
nodes[row][s * i + s - 1],
nodes[row + 1][s * i + Math.floor(s / 2) - 1],
"mediumvioletred"
).draw();
}
row++;
}

for (let i = 0; i < number_nodes; i++) {
new Arrow(
nodes[number_of_steps - 1][i],
output_nodes[i],
"darkcyan"
).draw();
}
}
// code block
break;
case "esact":
{
const indices = createIndices(number_nodes);
for (let i = 0; i < number_nodes; i++) {
new Arrow(
input_nodes[i],
nodes[0][indices[i]],
"darkcyan"
).draw();
nodes[0][i].addText(`X_{${indices[i]}}`);
}

let row = 0;
for (let s = number_nodes >> 1; s > 0; s >>= 1) {
for (let i = 0; i < s; i++) {
new Arrow(nodes[row][i + s], nodes[row + 1][i]).draw();
new Arrow(nodes[row][i], nodes[row + 1][i]).draw();
}
row++;
}
const sum_node = nodes[row++][0];
new Arrow(sum_node, nodes[row][0], "red").draw();
sum_node.elem.attr("fill", "lightcyan");
if (number_nodes > 1)
sum_node.addText(`\\sum_{i=0}^{${number_nodes - 1}}X_i`);
nodes[row][0].addText("0");

for (let s = 1; s < number_nodes; s <<= 1) {
for (let i = 0; i < s; i++) {
new Arrow(nodes[row][i + s], nodes[row + 1][i]).draw();
new Arrow(nodes[row][i], nodes[row + 1][i]).draw();
new Arrow(
nodes[row][i],
nodes[row + 1][i + s],
"mediumvioletred"
).draw();
}
row++;
}

for (let i = 0; i < number_nodes; i++) {
new Arrow(
nodes[row][indices[i]],
output_nodes[i],
"darkcyan"
).draw();
}
}
break;
default:
console.log("error");
}

scale_svg(number_nodes);

// svgD3.selectAll("foreignObject").each(function() {
//   MathJax.typeset([this]);
//   console.log(this)
// });
}

input.addEventListener("keypress", function (event) {
if (event.key === "Enter") {
event.preventDefault();
onApplyBtn();
}
});

function isPowerTwo(num) {
return Math.abs(2 ** Math.round(Math.log2(num)) - num) < 1e-3;
}

function onApplyBtn() {
const numnodes = parseInt(input.value);
if (isPowerTwo(numnodes)) {
create_svg(numnodes);
} else {
alert("Enter power of two");
}
}

onApplyBtn();

// var isTouchDevice = function () { return 'ontouchstart' in window || 'onmsgesturechange' in window; };
// var isDesktop = window.screenX != 0 && !isTouchDevice() ? true : false;

// console.log(isDesktop?"Desktop Mode": "Mobile Mode")
// if (isDesktop) {
addEventListener("resize", () => {
scale_svg(parseInt(input.value));
});
// }

function saveSvg(svgEl, name) {
svgEl.style = null;
svgEl.setAttribute("xmlns", "http://www.w3.org/2000/svg");
var svgData = svgEl.outerHTML;
var preface = '<?xml version="1.0" standalone="no"?>\r\n';
var svgBlob = new Blob([preface, svgData], {
type: "image/svg+xml;charset=utf-8",
});
var svgUrl = URL.createObjectURL(svgBlob);
var downloadLink = document.createElement("a");
downloadLink.href = svgUrl;
downloadLink.download = name;
document.body.appendChild(downloadLink);
downloadLink.click();
document.body.removeChild(downloadLink);
}

function onSaveBtn() {
saveSvg(svgElem.cloneNode(true), "graph.svg");
}

// });
