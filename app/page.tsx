'use client';
import { useState } from 'react';

export default function Home() {
  const [dato, setDato] = useState("");
  const [resultado, setResultado] = useState("");

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDato(event.target.value);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const response = await fetch(`/api/lala?dato=${dato}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" }, // Can be removed for GET requests
    });

    const data = await response.json();

    setResultado(data.resultado);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="text" name="dato" value={dato} onChange={handleChange} />
        <button type="submit">Enviar</button>
      </form>
      <p>{resultado}</p>
    </div>
  );
}
