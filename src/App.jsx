import Header from "./components/Header";
import AddForm from "./components/AddForm";
import Item from "./components/Item";
import { useState } from "react";
import "./App.css";

function App() {
  const [tasks, setTasks] = useState([
    { id: 1, title: "Fix bug program" },
    { id: 2, title: "Manual for programe" },
    { id: 3, title: "Going to bank" },
  ]);
  const [title, setTitle] = useState("");
  const [editId, setEditId] = useState(null);

  const deleteTask = (id) => {
    setTasks(tasks.filter((task) => task.id !== id));
  };

  const editTask = (id) => {
  const task = tasks.find((task) => task.id === id);
  setEditId(id);
  setTitle(task.title);
};
  const addTask = () => {
  if (title.trim() === "") return;
  if (editId !== null) {
    setTasks(tasks.map(task =>
      task.id === editId ? { ...task, title } : task
    ));
    setEditId(null);
  } else {
    setTasks([...tasks, { id: Date.now(), title }]);
  }
  setTitle("");
};

  return (
    <div className="App">
      <Header />
      <div className="container">
        <AddForm title={title} setTitle={setTitle} onAddTask={addTask} editId={editId} />
        <section>
          {tasks.map((data, id) => (
            <Item
              key={data.id}
              data={data}
              deleteTask={deleteTask}
              editTask={editTask}
              seq={id + 1}
            />
          ))}
        </section>
      </div>
    </div>
  );
}

export default App;
