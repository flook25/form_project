import "./AddForm.css";

export default function AddForm(props) {
  const { title, setTitle, onAddTask, editId } = props;

  const handleSubmit = (e) => {
    e.preventDefault();
    onAddTask();
  };

  return (
    <>
      <h2>Form for managing the tasks</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-control">
          <input
            type="text"
            className="text-input"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <button type="submit" className="submit-btn">
            {editId ? "update" : "add"}
          </button>
        </div>
      </form>
    </>
  );
}