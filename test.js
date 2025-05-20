const input  = document.getElementById('taskInput');
const addBtn = document.getElementById('addBtn');
const list   = document.getElementById('taskList');

addBtn.addEventListener('click', () => {
  const text = input.value.trim();
  if (!text) return;
  
  const li = document.createElement('li');
  li.textContent = text;
  
  // toggle “completed” on click
  li.addEventListener('click', () => {
    li.classList.toggle('completed');
  });
  
  list.appendChild(li);
  input.value = '';
});
