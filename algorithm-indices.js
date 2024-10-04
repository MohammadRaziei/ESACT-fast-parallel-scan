function copyArrayToClipboard(arr) {
const textToCopy = arr.join(","); // Join the array elements with a comma
navigator.clipboard
.writeText(textToCopy)
.then(() => {
console.log("Array copied to clipboard");
})
.catch((err) => {
console.error("Failed to copy text: ", err);
});
}
// Function to create a table from an array
function createTableFromArray(arr) {
const table = document.createElement("table");
table.classList.add(
"table-auto",
"w-40",
"text-left",
"shadow-md",
"bg-white",
"content-start"
);

// Create header row
const thead = table.createTHead();
thead.classList.add(
"bg-gray-800",
"text-white",
"content-center",
"h-10"
);
const headerRow = thead.insertRow();
const thIndex = document.createElement("th");
thIndex.classList.add("px-4");
thIndex.textContent = "Index";
const thValue = document.createElement("th");
thValue.classList.add("px-4");
thValue.textContent = "Value";
headerRow.appendChild(thIndex);
headerRow.appendChild(thValue);

// Create body of the table
const tbody = table.createTBody();
arr.forEach((value, index) => {
const row = tbody.insertRow();
const cellIndex = row.insertCell();
const cellValue = row.insertCell();
cellIndex.textContent = index;
cellValue.textContent = value;
cellIndex.classList.add("p-3");
cellValue.classList.add("p-3");
});

return table;
}

function onApplyArrayBtn() {
const numnodes = document.getElementById("inputArray").value;
if (isPowerTwo(numnodes)) {
const tableElem = document.getElementById("array-table");
while (tableElem.firstChild) {
tableElem.removeChild(tableElem.firstChild);
}
tableElem.appendChild(createTableFromArray(createIndices(numnodes)));
} else {
alert("Enter power of two");
}
}
// Append the table to the div

function onCopyBtn() {
copyArrayToClipboard(
    createIndices(document.getElementById("inputArray").value)
);
}

onApplyArrayBtn();
